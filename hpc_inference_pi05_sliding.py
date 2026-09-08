"""
Sliding-window inference server for Pi0.5.

Differences from hpc_inference_pi05.py
---------------------------------------
- Inference is triggered every `execution_horizon` actions served (not when buffer empties).
- When new chunk arrives it calls overwrite() — discarding leftover old actions immediately.
- While inference is running the robot keeps executing the current buffer, so it never stalls.

Timeline (chunk_size=50, execution_horizon=16, control_freq=30 Hz):
  t=0          : first obs → inference → buffer filled with 50 actions
  t=16 actions : trigger next inference (still 34 actions left in buffer)
  t=??         : inference done (~500ms later, ~15 steps into it) → overwrite buffer
                 seamless: robot was already executing from old buffer
  ...repeats every 16 steps
"""

import zmq
import numpy as np
import time
import torch
import argparse
import threading
import sys
from pi05_inference import Pi05InferencePolicy


# ---------------------------------------------------------------------------
# ActionBuffer  (identical to hpc_inference_pi05.py)
# ---------------------------------------------------------------------------

class ActionBuffer:
    """Thread-safe buffer that can be overwritten with new predictions."""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.actions = []
        self.lock = threading.Lock()
        self.last_update_time = None
        self.update_count = 0
        self.total_pops = 0
        self.wrapped_pops = 0
        self.chunk_metadata = None

    def overwrite(self, action_chunk, metadata=None):
        """Replace buffer contents unconditionally."""
        with self.lock:
            self.actions = list(action_chunk)
            self.last_update_time = time.time()
            self.update_count += 1
            self.chunk_metadata = metadata

    def empty_overwrite(self, action_chunk):
        """Write only if buffer is currently empty (kept for compatibility)."""
        with self.lock:
            if len(self.actions) == 0:
                self.actions = list(action_chunk)
                self.last_update_time = time.time()
                self.update_count += 1

    def pop_action(self):
        with self.lock:
            if not self.actions:
                return None
            action = self.actions.pop(0)
            self.total_pops += 1
            buffer_age = time.time() - self.last_update_time if self.last_update_time else 0
            action['buffer_remaining'] = len(self.actions)
            action['buffer_age'] = buffer_age
            action['total_buffer_updates'] = self.update_count
            action['is_stale'] = self.total_pops > self.update_count * self.chunk_size
            if action['is_stale']:
                self.wrapped_pops += 1
            return action

    def pop_all(self):
        with self.lock:
            if not self.actions:
                return None, None
            actions = self.actions
            self.actions = []
            self.total_pops += len(actions)
            metadata = self.chunk_metadata
            self.chunk_metadata = None
            return actions, metadata

    def get_status(self):
        with self.lock:
            return {
                'actions_remaining': len(self.actions),
                'chunk_size': self.chunk_size,
                'update_count': self.update_count,
                'total_pops': self.total_pops,
                'wrapped_pops': self.wrapped_pops,
                'last_update_time': self.last_update_time,
                'age': time.time() - self.last_update_time if self.last_update_time else None,
            }

    @property
    def is_empty(self):
        with self.lock:
            return len(self.actions) == 0

    def clear(self):
        with self.lock:
            self.actions = []
            print("ActionBuffer cleared")


# ---------------------------------------------------------------------------
# SlidingInferenceServer
# ---------------------------------------------------------------------------

class SlidingInferenceServer:
    def __init__(
        self,
        model,
        obs_port=5555,
        action_port=5556,
        device='cuda',
        chunk_size=50,
        execution_horizon=16,
        use_async_rtc=False,
    ):
        """
        Args:
            model:              Trained policy (Pi05InferencePolicy or compatible).
            obs_port:           ZMQ port for receiving observations (SUB).
            action_port:        ZMQ port for serving actions (REP).
            chunk_size:         Full prediction horizon H (model output length).
            execution_horizon:  How many actions to execute before triggering the
                                next inference.  Must be < chunk_size so there are
                                still actions in the buffer while inference runs.
            use_async_rtc:      Use async RTC inference path.
        """
        assert execution_horizon < chunk_size, (
            f"execution_horizon ({execution_horizon}) must be < chunk_size ({chunk_size})"
        )

        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.use_async_rtc = use_async_rtc

        self.action_buffer = ActionBuffer(chunk_size)

        # ZMQ
        self.context = zmq.Context()

        self.obs_socket = self.context.socket(zmq.SUB)
        self.obs_socket.bind(f"tcp://*:{obs_port}")
        self.obs_socket.subscribe(b"")
        self.obs_socket.setsockopt(zmq.CONFLATE, 1)  # keep only latest

        self.action_socket = self.context.socket(zmq.REP)
        self.action_socket.bind(f"tcp://*:{action_port}")

        self.control_port = action_port + 1
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{self.control_port}")

        print(f"Observation receiver  : port {obs_port}", flush=True)
        print(f"Action server         : port {action_port}", flush=True)
        print(f"Control server        : port {self.control_port}", flush=True)
        print(f"Chunk size (H)        : {chunk_size}", flush=True)
        print(f"Execution horizon     : {execution_horizon}", flush=True)
        print(f"Safety margin         : {chunk_size - execution_horizon} actions "
              f"({(chunk_size - execution_horizon) / 30 * 1000:.0f} ms at 30 Hz)", flush=True)

        print("\nWARMUP: waiting for torch.compile ...", flush=True)
        self.model.warmup()
        print("Warmup complete.", flush=True)

        # Threading
        self.stop_event = threading.Event()

        # Inference trigger: set by action-serving loop every execution_horizon pops.
        # Also fires on startup (buffer empty) so the very first chunk is generated.
        self._infer_trigger = threading.Event()
        self._infer_trigger.set()          # arm immediately so first obs triggers inference
        self._actions_since_trigger = 0    # protected by _trigger_lock
        self._trigger_lock = threading.Lock()

        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True)
        self.action_thread = threading.Thread(
            target=self._action_serving_loop, daemon=True)
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True)

        self.stats = {
            'observations_received': 0,
            'inferences_completed': 0,
            'actions_served': 0,
            'total_inference_time': 0.0,
            'buffer_overwrites': 0,
            'stale_actions': 0,
            'errors': 0,
        }
        self.stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self):
        print("Inference thread started (sliding-window mode)", flush=True)

        while not self.stop_event.is_set():
            try:
                # Drain obs socket, keep only the latest
                observation = None
                while self.obs_socket.poll(timeout=0):
                    observation = self.obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    if observation is not None:
                        with self.stats_lock:
                            self.stats['observations_received'] += 1

                # Trigger condition:
                #   • _infer_trigger is set (every execution_horizon pops, or startup)
                #   • AND we actually have a fresh observation
                if observation is not None and self._infer_trigger.is_set():
                    self._infer_trigger.clear()

                    print("--calling inference--", flush=True)
                    t0 = time.time()
                    action_chunk, metadata = self._run_inference_chunk(observation)
                    inference_time = time.time() - t0

                    # Always overwrite — discard any leftover old actions
                    if self.use_async_rtc:
                        self.action_buffer.overwrite(action_chunk, metadata)
                    else:
                        self.action_buffer.overwrite(action_chunk)

                    with self.stats_lock:
                        self.stats['inferences_completed'] += 1
                        self.stats['total_inference_time'] += inference_time
                        self.stats['buffer_overwrites'] += 1

                    n = self.stats['inferences_completed']
                    avg_ms = self.stats['total_inference_time'] / n * 1000
                    print(
                        f"Inference #{n}: {inference_time*1000:.1f}ms "
                        f"(avg {avg_ms:.1f}ms)  "
                        f"buffer_remaining={self.action_buffer.get_status()['actions_remaining']}",
                        flush=True,
                    )

            except zmq.Again:
                continue
            except Exception as e:
                print(f"Inference error: {e}", flush=True)
                import traceback; traceback.print_exc()
                with self.stats_lock:
                    self.stats['errors'] += 1
                time.sleep(0.1)

        print("Inference thread stopped")

    # ------------------------------------------------------------------
    # Action serving loop
    # ------------------------------------------------------------------

    def _action_serving_loop(self):
        print("Action serving thread started (sliding-window mode)", flush=True)

        while not self.stop_event.is_set():
            try:
                self.action_socket.recv_pyobj()  # block until robot requests

                if self.use_async_rtc:
                    action, metadata = self.action_buffer.pop_all()
                else:
                    action = self.action_buffer.pop_action()

                if action is None:
                    response = {
                        'error': 'buffer_empty',
                        'message': 'Buffer empty — inference may be overdue',
                    }
                    print("WARNING: Buffer empty! "
                          "Inference is slower than execution_horizon steps.", flush=True)
                else:
                    response = action if not self.use_async_rtc else {
                        "chunk": action, "metadata": metadata
                    }

                    if not self.use_async_rtc:
                        # Count pops and fire trigger every execution_horizon steps
                        with self._trigger_lock:
                            self._actions_since_trigger += 1
                            if self._actions_since_trigger >= self.execution_horizon:
                                self._actions_since_trigger = 0
                                self._infer_trigger.set()
                                print(
                                    f"[trigger] {self.execution_horizon} actions served — "
                                    f"requesting next inference  "
                                    f"(buf_remaining={action.get('buffer_remaining', '?')})",
                                    flush=True,
                                )

                        if action.get('is_stale', False):
                            with self.stats_lock:
                                self.stats['stale_actions'] += 1

                self.action_socket.send_pyobj(response)

                with self.stats_lock:
                    self.stats['actions_served'] += 1
                    if self.stats['actions_served'] % 150 == 0:
                        self._print_stats()

            except Exception as e:
                print(f"Action serving error: {e}", flush=True)
                import traceback; traceback.print_exc()
                raise

        print("Action serving thread stopped")

    # ------------------------------------------------------------------
    # Control loop (identical to original)
    # ------------------------------------------------------------------

    def _control_loop(self):
        print("Control thread started", flush=True)
        while not self.stop_event.is_set():
            try:
                command = self.control_socket.recv_pyobj()
                print(f"Control command: {command}", flush=True)

                if command.get('command') == 'clear_queue':
                    self.action_buffer.clear()
                    with self._trigger_lock:
                        self._actions_since_trigger = 0
                        self._infer_trigger.set()   # re-arm so next obs triggers inference
                    try:
                        self.model.reset_action_queue()
                        print("Policy action queue reset")
                    except Exception as e:
                        print(f"Warning: could not reset policy queue: {e}")
                    response = {'status': 'ok', 'message': 'Queue cleared'}
                else:
                    response = {'status': 'error',
                                'message': f'Unknown command: {command.get("command")}'}

                self.control_socket.send_pyobj(response)

            except Exception as e:
                print(f"Control thread error: {e}", flush=True)
                import traceback; traceback.print_exc()

        print("Control thread stopped")

    # ------------------------------------------------------------------
    # Inference helpers (identical to original)
    # ------------------------------------------------------------------

    def preprocess_observation(self, observation):
        return observation  # Pi05InferencePolicy handles raw dicts

    def postprocess_action_chunk(self, model_output, observation=None):
        if isinstance(model_output, torch.Tensor):
            output = model_output.detach().cpu().numpy()
        else:
            output = model_output
        if output.ndim == 3:
            output = output.squeeze(0)

        action_chunk = []
        for t in range(output.shape[0]):
            a = output[t]
            if self.model.is_delta_action:
                d = {'left_delta_pose': a[0:7], 'right_delta_pose': a[7:14]}
            else:
                d = {'left_ee_pose': a[0:7], 'right_ee_pose': a[7:14]}
            d['left_gripper'] = float(a[14])
            d['right_gripper'] = float(a[15])
            d['timestamp'] = time.time()
            d['chunk_index'] = t
            if observation is not None:
                d['obs_timestamp'] = observation.get('timestamp')
            action_chunk.append(d)
        return action_chunk

    def _run_inference_chunk(self, observation):
        model_input = self.preprocess_observation(observation)
        if self.use_async_rtc:
            with torch.no_grad():
                model_output, metadata = self.model.predict_action_chunk_async(
                    model_input, transform_to_quat=True
                )
            action_chunk = self.postprocess_action_chunk(model_output, observation)
            return action_chunk, metadata
        else:
            with torch.no_grad():
                model_output = self.model.forward(model_input)
            action_chunk = self.postprocess_action_chunk(model_output, observation)
            return action_chunk, None

    # ------------------------------------------------------------------
    # Stats / lifecycle
    # ------------------------------------------------------------------

    def _print_stats(self):
        with self.stats_lock:
            s = self.stats.copy()
        buf = self.action_buffer.get_status()
        print(f"\n{'='*60}")
        print(f"Stats ({s['actions_served']} actions served):")
        print(f"  Observations received : {s['observations_received']}")
        print(f"  Inferences completed  : {s['inferences_completed']}")
        print(f"  Buffer overwrites     : {s['buffer_overwrites']}")
        print(f"  Stale actions served  : {s['stale_actions']}")
        print(f"  Errors                : {s['errors']}")
        if s['inferences_completed'] > 0:
            avg = s['total_inference_time'] / s['inferences_completed']
            print(f"  Avg inference time    : {avg*1000:.1f} ms")
            print(f"  Inference frequency   : {1/avg:.2f} Hz")
            safety = (self.chunk_size - self.execution_horizon) / 30 * 1000
            print(f"  Safety margin         : {safety:.0f} ms "
                  f"({'OK' if avg*1000 < safety else 'TIGHT — consider larger chunk'})")
        print(f"  Buffer remaining      : {buf['actions_remaining']}/{buf['chunk_size']}")
        print(f"{'='*60}\n")

    def run(self):
        print("\nStarting SlidingInferenceServer ...")
        print(f"  Strategy: infer every {self.execution_horizon} steps, "
              f"overwrite buffer on arrival\n")
        self.inference_thread.start()
        self.action_thread.start()
        self.control_thread.start()
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down ...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_event.set()
        self.inference_thread.join(timeout=2.0)
        self.action_thread.join(timeout=2.0)
        self.control_thread.join(timeout=2.0)
        print("\nFinal stats:")
        self._print_stats()
        self.obs_socket.close()
        self.action_socket.close()
        self.control_socket.close()
        self.context.term()
        print("Server stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_policy_model(checkpoint_path, args, device='cuda'):
    return Pi05InferencePolicy(
        checkpoint_path=checkpoint_path,
        device=device,
        primary_camera='cam_high',
        control_freq=args.hz,
        inference_delay=args.inference_delay,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Sliding-window HPC Inference Server (overlapping chunks)'
    )
    parser.add_argument('--obs-port',          type=int,   default=5555)
    parser.add_argument('--action-port',        type=int,   default=5556)
    parser.add_argument('--checkpoint',         type=str,   default=None)
    parser.add_argument('--device',             type=str,   default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--pred-horizon',       type=int,   default=50,
                        help='Full chunk size H (default: 50)')
    parser.add_argument('--execution-horizon',  type=int,   default=16,
                        help='Steps between inferences (default: 16); must be < pred-horizon')
    parser.add_argument('--inference-delay',    type=int,   default=3,
                        help='Inference delay in action steps (for async RTC)')
    parser.add_argument('--hz',                 type=float, default=30.0)
    parser.add_argument('--use-async-rtc',      action='store_true')
    args = parser.parse_args()

    if args.execution_horizon >= args.pred_horizon:
        parser.error(
            f"--execution-horizon ({args.execution_horizon}) must be "
            f"< --pred-horizon ({args.pred_horizon})"
        )

    print(f"Loading policy from {args.checkpoint} ...", flush=True)
    model = load_policy_model(args.checkpoint, args, device=args.device)

    server = SlidingInferenceServer(
        model,
        obs_port=args.obs_port,
        action_port=args.action_port,
        device=args.device,
        chunk_size=args.pred_horizon,
        execution_horizon=args.execution_horizon,
        use_async_rtc=args.use_async_rtc,
    )
    server.run()


if __name__ == "__main__":
    main()
