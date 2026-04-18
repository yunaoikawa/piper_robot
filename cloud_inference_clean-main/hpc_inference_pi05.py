import zmq
import numpy as np
import time
import torch
import argparse
import threading
import sys
from scipy.spatial.transform import Rotation as R
from pi05_inference import Pi05InferencePolicy


def r6_absolute_to_quat(r6: np.ndarray) -> np.ndarray:
    """Convert absolute 6D rotation to quaternion (wxyz). Always returns (B, 4)."""
    if r6.ndim == 1:
        r6 = r6[None]
    r1 = r6[..., :3]
    r2 = r6[..., 3:6]
    b1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    b2 = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)
    quat = R.from_matrix(mat).as_quat(scalar_first=True)
    if quat.ndim == 1:
        quat = quat[None, :]
    return quat


def r20_to_quat16(action: np.ndarray) -> np.ndarray:
    """
    Convert r6 action (20D) to quaternion action (16D).

    Input:  (20,) or (B, 20) or (1, B, 20)
        [left_pos(3), left_r6(6), left_grip(1), right_pos(3), right_r6(6), right_grip(1)]
    Output: (B, 16)
        [left_quat(4), left_pos(3), right_quat(4), right_pos(3), left_grip(1), right_grip(1)]
    """
    if action.ndim == 3:
        action = action.squeeze(0)
    if action.ndim == 1:
        action = action[None, :]

    B = action.shape[0]
    out = np.zeros((B, 16), dtype=np.float32)

    out[:, 0:4] = r6_absolute_to_quat(action[:, 3:9])
    out[:, 4:7] = action[:, 0:3]
    out[:, 7:11] = r6_absolute_to_quat(action[:, 13:19])
    out[:, 11:14] = action[:, 10:13]
    out[:, 14] = action[:, 9]
    out[:, 15] = action[:, 19]

    return out


class ActionBuffer:
    """Thread-safe buffer that gets completely overwritten with new predictions."""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.actions = []
        self.lock = threading.Lock()
        self.last_update_time = None
        self.update_count = 0
        self.total_pops = 0
        self.wrapped_pops = 0

    def overwrite(self, action_chunk):
        """Completely overwrite buffer with new action chunk."""
        with self.lock:
            self.actions = list(action_chunk)
            self.last_update_time = time.time()
            self.update_count += 1

    def pop_action(self):
        """Pop and return the first action from buffer."""
        with self.lock:
            if len(self.actions) == 0:
                return None

            action = self.actions.pop(0)
            self.total_pops += 1

            if action is not None:
                buffer_age = time.time() - self.last_update_time if self.last_update_time else 0
                action['buffer_remaining'] = len(self.actions)
                action['buffer_age'] = buffer_age
                action['total_buffer_updates'] = self.update_count

                if self.total_pops > self.update_count * self.chunk_size:
                    self.wrapped_pops += 1
                    action['is_stale'] = True
                else:
                    action['is_stale'] = False

            return action

    def get_status(self):
        with self.lock:
            return {
                'actions_remaining': len(self.actions),
                'chunk_size': self.chunk_size,
                'update_count': self.update_count,
                'total_pops': self.total_pops,
                'wrapped_pops': self.wrapped_pops,
                'last_update_time': self.last_update_time,
                'age': time.time() - self.last_update_time if self.last_update_time else None
            }

    @property
    def is_empty(self):
        with self.lock:
            return len(self.actions) == 0

    def clear(self):
        with self.lock:
            self.actions = []
            print("ActionBuffer cleared")


class InferenceServer:
    def __init__(self, model, obs_port=5555, action_port=5556, device='cuda', chunk_size=50):
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.action_buffer = ActionBuffer(chunk_size)

        # ZeroMQ setup
        self.context = zmq.Context()

        self.obs_socket = self.context.socket(zmq.SUB)
        self.obs_socket.bind(f"tcp://*:{obs_port}")
        self.obs_socket.subscribe(b"")
        self.obs_socket.setsockopt(zmq.CONFLATE, 1)

        self.action_socket = self.context.socket(zmq.REP)
        self.action_socket.bind(f"tcp://*:{action_port}")

        self.control_port = action_port + 1
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{self.control_port}")

        print(f"Observation receiver listening on port {obs_port}", flush=True)
        print(f"Action server listening on port {action_port}", flush=True)
        print(f"Control server listening on port {self.control_port}", flush=True)
        print(f"Using device: {device}", flush=True)
        print(f"Action chunk size: {chunk_size}", flush=True)

        print("Warming up model...", flush=True)
        self.model.warmup()
        print("Warmup complete. Inference server is ready.", flush=True)

        # Threading
        self.stop_event = threading.Event()
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.action_thread = threading.Thread(target=self._action_serving_loop, daemon=True)
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)

        self.stats = {
            'observations_received': 0,
            'inferences_completed': 0,
            'actions_served': 0,
            'total_inference_time': 0.0,
            'buffer_wraps': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()

    def _inference_loop(self):
        """Continuously receive observations and run inference."""
        print("Inference thread started", flush=True)

        while not self.stop_event.is_set():
            try:
                # Drain observation socket, keep only latest
                observation = None
                while self.obs_socket.poll(timeout=0):
                    observation = self.obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    if observation is not None and self.action_buffer.is_empty:
                        with self.stats_lock:
                            self.stats['observations_received'] += 1

                if observation is not None and self.action_buffer.is_empty:
                    start_time = time.time()

                    # Run inference: get raw r6 action chunk (20D)
                    obs_input = observation
                    with torch.no_grad():
                        raw_chunk = self.model.predict_action_chunk(
                            obs_input, transform_to_quat=False
                        )

                    # Convert to numpy
                    if isinstance(raw_chunk, torch.Tensor):
                        raw_chunk = raw_chunk.detach().cpu().numpy()
                    elif isinstance(raw_chunk, dict) and "action" in raw_chunk:
                        raw_chunk = raw_chunk["action"]
                        if isinstance(raw_chunk, torch.Tensor):
                            raw_chunk = raw_chunk.detach().cpu().numpy()

                    # Convert r6 (20D) -> quat (16D)
                    quat_chunk = r20_to_quat16(raw_chunk)  # (B, 16)

                    inference_time = time.time() - start_time

                    # Build action dicts
                    action_list = []
                    for t in range(quat_chunk.shape[0]):
                        a = quat_chunk[t]
                        action = {
                            'left_ee_pose': a[0:7],
                            'right_ee_pose': a[7:14],
                            'left_gripper': float(a[14]),
                            'right_gripper': float(a[15]),
                            'timestamp': time.time(),
                            'chunk_index': t,
                            'obs_timestamp': observation.get('timestamp', None),
                        }
                        action_list.append(action)

                    # Overwrite buffer (always replace old actions)
                    self.action_buffer.overwrite(action_list)

                    with self.stats_lock:
                        self.stats['inferences_completed'] += 1
                        self.stats['total_inference_time'] += inference_time

                    if self.stats['inferences_completed'] % 10 == 0:
                        avg = self.stats['total_inference_time'] / self.stats['inferences_completed']
                        remaining = self.action_buffer.get_status()['actions_remaining']
                        print(f"Inference #{self.stats['inferences_completed']}: "
                              f"{inference_time*1000:.0f}ms (avg: {avg*1000:.0f}ms), "
                              f"chunk={quat_chunk.shape[0]}, buffer={remaining}",
                              flush=True)

            except zmq.Again:
                continue
            except Exception as e:
                print(f"Inference error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                with self.stats_lock:
                    self.stats['errors'] += 1
                time.sleep(0.1)

        print("Inference thread stopped")

    def _action_serving_loop(self):
        """Serve actions from buffer on demand."""
        print("Action serving thread started", flush=True)

        while not self.stop_event.is_set():
            try:
                request = self.action_socket.recv_pyobj()

                action = self.action_buffer.pop_action()

                if action is None:
                    response = {
                        'error': 'buffer_empty',
                        'message': 'Buffer drained, waiting for inference'
                    }
                else:
                    response = action

                    if action.get('is_stale', False):
                        with self.stats_lock:
                            self.stats['buffer_wraps'] += 1
                        if self.stats['buffer_wraps'] % 50 == 1:
                            print(f"WARNING: Serving stale actions (total: {self.stats['buffer_wraps']})")

                    remaining = action.get('buffer_remaining', 0)
                    if remaining == 2:
                        print(f"Buffer running low: {remaining} actions remaining")

                self.action_socket.send_pyobj(response)

                with self.stats_lock:
                    self.stats['actions_served'] += 1

                if self.stats['actions_served'] % 100 == 0:
                    self._print_stats()

            except Exception as e:
                print(f"Action serving error: {e}", flush=True)
                with self.stats_lock:
                    self.stats['errors'] += 1
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise e

        print("Action serving thread stopped")

    def _control_loop(self):
        """Listen for control commands (e.g., clear queue)."""
        print("Control thread started", flush=True)

        while not self.stop_event.is_set():
            try:
                command = self.control_socket.recv_pyobj()
                print(f"Received control command: {command}", flush=True)

                if command.get('command') == 'clear_queue':
                    self.action_buffer.clear()
                    try:
                        self.model.reset_action_queue()
                        print("Policy action queue reset")
                    except Exception as e:
                        print(f"Warning: Could not reset policy queue: {e}")
                    response = {'status': 'ok', 'message': 'Action queue cleared'}
                else:
                    response = {'status': 'error', 'message': f'Unknown command'}

                self.control_socket.send_pyobj(response)

            except Exception as e:
                print(f"Control thread error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        print("Control thread stopped")

    def _print_stats(self):
        with self.stats_lock:
            stats = self.stats.copy()
        buf = self.action_buffer.get_status()

        print(f"\n{'='*60}")
        print(f"Stats after {stats['actions_served']} actions served:")
        print(f"  Observations received: {stats['observations_received']}")
        print(f"  Inferences completed: {stats['inferences_completed']}")
        print(f"  Actions served: {stats['actions_served']}")
        if stats['inferences_completed'] > 0:
            avg = stats['total_inference_time'] / stats['inferences_completed']
            print(f"  Avg inference time: {avg*1000:.1f}ms ({1.0/avg:.1f} Hz)")
        print(f"  Stale actions: {stats['buffer_wraps']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Buffer: {buf['actions_remaining']}/{buf['chunk_size']}")
        if buf['age'] is not None:
            print(f"  Buffer age: {buf['age']*1000:.0f}ms")
        print(f"{'='*60}\n")

    def run(self):
        print("\nStarting inference server...")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Buffer strategy: always overwrite\n")

        self.inference_thread.start()
        self.action_thread.start()
        self.control_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_event.set()
        self.inference_thread.join(timeout=2.0)
        self.action_thread.join(timeout=2.0)
        self.control_thread.join(timeout=2.0)
        print("\nFinal Statistics:")
        self._print_stats()
        self.obs_socket.close()
        self.action_socket.close()
        self.control_socket.close()
        self.context.term()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description='HPC Inference Server for Pi0.5')
    parser.add_argument('--obs-port', type=int, default=5555)
    parser.add_argument('--action-port', type=int, default=5556)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--pred_horizon', default=50, type=int, help='Action chunk size')
    parser.add_argument('--inference-delay', default=3, type=int)
    parser.add_argument('--hz', default=30, type=float, help='Control frequency')
    args = parser.parse_args()

    print(f"Loading policy model from {args.checkpoint}...", flush=True)
    model = Pi05InferencePolicy(
        is_delta_action=False,
        checkpoint_path=args.checkpoint,
        device=args.device,
        primary_camera='cam_high',
        control_freq=args.hz,
        inference_delay=args.inference_delay,
    )

    print(f"Server configuration:", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Prediction horizon: {args.pred_horizon}", flush=True)
    print(f"  Control frequency: {args.hz} Hz", flush=True)

    server = InferenceServer(
        model,
        obs_port=args.obs_port,
        action_port=args.action_port,
        device=args.device,
        chunk_size=args.pred_horizon,
    )
    server.run()


if __name__ == "__main__":
    main()