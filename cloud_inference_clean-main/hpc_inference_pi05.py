import zmq
import numpy as np
import time
import torch
import argparse
import threading
# from scipy.spatial.transform import Rotation as R
import sys
from pi05_inference import Pi05InferencePolicy

class ActionBuffer:
    """Thread-safe buffer that gets completely overwritten with new predictions."""
    
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.actions = []  # Start empty
        self.lock = threading.Lock()
        self.last_update_time = None
        self.update_count = 0
        self.total_pops = 0
        self.wrapped_pops = 0
        self.chunk_metadata = None  # Store metadata from last inference, if using async RTC
        
    def overwrite(self, action_chunk, metadata=None):
        """
        Completely overwrite buffer with new action chunk.
        
        Args:
            action_chunk: List of action dicts
        """
        with self.lock:
            self.actions = list(action_chunk)  # Make a copy
            self.last_update_time = time.time()
            self.update_count += 1
            self.chunk_metadata = metadata

    def empty_overwrite(self, action_chunk):
        """
        Overwrite buffer only if it's currently empty.
        
        Args:
            action_chunk: List of action dicts
        """
        with self.lock:
            if len(self.actions) == 0:
                self.actions = list(action_chunk)
                self.last_update_time = time.time()
                self.update_count += 1
            
    def pop_action(self):
        """
        Pop and return the first action from buffer.
        After popping A0, buffer becomes [A1, A2, ..., An].
        
        Returns:
            Action dict, or None if buffer is empty
        """
        with self.lock:
            if len(self.actions) == 0:
                return None
            
            # Pop the first action
            action = self.actions.pop(0)
            self.total_pops += 1
            
            # Check if we're reusing old actions (wrapped around)
            # This happens if buffer was already empty and got refilled
            # We detect this by checking if action is stale
            if action is not None:
                buffer_age = time.time() - self.last_update_time if self.last_update_time else 0
                action_age = buffer_age + (self.total_pops - self.update_count * self.chunk_size) / 30.0
                
                # Add metadata
                action['buffer_remaining'] = len(self.actions)
                action['buffer_age'] = buffer_age
                action['total_buffer_updates'] = self.update_count
                
                # Track if we popped more actions than we've generated
                if self.total_pops > self.update_count * self.chunk_size:
                    self.wrapped_pops += 1
                    action['is_stale'] = True
                else:
                    action['is_stale'] = False
            
            return action
        
    def pop_all(self):
        """Pop and return all actions from buffer."""
        with self.lock:
            if len(self.actions) == 0:
                return None, None
            actions = self.actions
            self.actions = []
            self.total_pops += len(actions)
            metadata = self.chunk_metadata
            self.chunk_metadata = None
            return actions, metadata
    
    def get_status(self):
        """Get buffer status for monitoring."""
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
        """Clear all actions from buffer."""
        with self.lock:
            self.actions = []
            print("ActionBuffer cleared")


class InferenceServer:
    def __init__(self, model, obs_port=5555, action_port=5556, device='cuda', chunk_size=10, use_async_rtc=False):
        """
        HPC inference server with asynchronous inference and action serving.

        Args:
            model: Your trained policy model
            obs_port: Port for receiving observations from robot
            action_port: Port for serving actions to robot
            device: 'cuda' or 'cpu'
            chunk_size: Action buffer size (should be H = full prediction horizon)
            use_async_rtc: Whether to use async RTC inference
        """
        self.model = model
        self.device = device
        # self.model.to(device)
        # self.model.eval()
        # unnecessary because the Policy class alredy handles this

        self.chunk_size = chunk_size
        self.use_async_rtc = use_async_rtc
        self.action_buffer = ActionBuffer(chunk_size)
        
        # ZeroMQ setup - two separate sockets
        self.context = zmq.Context()
        
        # Socket 1: Receive observations (SUB pattern - non-blocking)
        self.obs_socket = self.context.socket(zmq.SUB)
        self.obs_socket.bind(f"tcp://*:{obs_port}")
        self.obs_socket.subscribe(b"")
        self.obs_socket.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message
        
        # Socket 2: Serve actions (REP pattern - blocking)
        self.action_socket = self.context.socket(zmq.REP)
        self.action_socket.bind(f"tcp://*:{action_port}")

        # Socket 3: Control commands (REP pattern - blocking)
        self.control_port = action_port + 1
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{self.control_port}")
        
        print(f"Observation receiver listening on port {obs_port}", flush=True)
        print(f"Action server listening on port {action_port}", flush=True)
        print(f"Control server listening on port {self.control_port}", flush=True)
        print(f"Using device: {device}", flush=True)

        print("IMPORTANT: As Pi0.5 is being run with torch.compile, additional time is needed for warmup. Please wait...")
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
        """
        Continuously receive observations and run inference.
        Overwrites buffer with each new prediction.
        """
        print("Inference thread started - continuously running inference", flush=True)
        
        while not self.stop_event.is_set():
            try:
                # Non-blocking receive of latest observation
                # if self.obs_socket.poll(timeout=100):  # 100ms timeout
                #     observation = self.obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    
                #     with self.stats_lock:
                #         self.stats['observations_received'] += 1
                #         print(f"Received observation #{self.stats['observations_received']}", flush=True)
                observation = None
                while self.obs_socket.poll(timeout=0):
                    # print("polling for obs succeeded...", flush=True)
                    observation = self.obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    if observation is not None:
                        with self.stats_lock:
                            self.stats['observations_received'] += 1
                            print(f"Received observation #{self.stats['observations_received']}", flush=True)
                if observation is not None and self.action_buffer.is_empty: # TODO remove is_empty check if temporal ensembling
                    # Run inference
                    print("--calling inference--", flush=True)
                    start_time = time.time()
                    action_chunk, metadata = self._run_inference_chunk(observation)
                    # action_chunk = self._run_inference_single(observation)
                    inference_time = time.time() - start_time

                    print(f"DEBUG: Inference completed in {inference_time*1000:.1f}ms", flush=True)
                    print(f"DEBUG: Action chunk: {action_chunk}", flush=True)
                    
                    # Overwrite buffer with new predictions
                    if self.use_async_rtc:
                        self.action_buffer.overwrite(action_chunk, metadata)
                    else:
                        self.action_buffer.empty_overwrite(action_chunk)
                    
                    with self.stats_lock:
                        self.stats['inferences_completed'] += 1
                        self.stats['total_inference_time'] += inference_time
                    
                    # Periodic logging
                    if self.stats['inferences_completed'] % 10 == 0:
                        avg_time = self.stats['total_inference_time'] / self.stats['inferences_completed']
                        print(f"Inference #{self.stats['inferences_completed']}: "
                                f"{inference_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
                        
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
        """
        Serve actions at constant rate by popping from buffer.
        Uses REQ-REP pattern for synchronization with robot.
        """
        print("Action serving thread started", flush=True)
        
        print("="*60, flush=True)
        print("ACTION THREAD DIAGNOSTICS", flush=True)
        print("="*60, flush=True)
        
        # Get socket info
        print(f"Socket type: {self.action_socket.socket_type} (should be {zmq.REP})", flush=True)
        print(f"Socket FD: {self.action_socket.getsockopt(zmq.FD)}", flush=True)
        print(f"RCVHWM: {self.action_socket.getsockopt(zmq.RCVHWM)}", flush=True)
        print(f"SNDHWM: {self.action_socket.getsockopt(zmq.SNDHWM)}", flush=True)
        
        # Check events available
        print(f"Initial events: {self.action_socket.getsockopt(zmq.EVENTS)}", flush=True)
        print("="*60, flush=True)

        while not self.stop_event.is_set():
            try:
                # Wait for action request from robot (blocking)
                request = self.action_socket.recv_pyobj()
                # print("Received action request from robot", flush=True)
                
                # Pop next action from buffer
                if self.use_async_rtc:
                    action, metadata = self.action_buffer.pop_all() # action here is actually a chunk
                else:
                    action = self.action_buffer.pop_action()
                
                if action is None:
                    # Buffer empty - waiting for inference
                    response = {
                        'error': 'buffer_empty',
                        'message': 'Buffer drained, waiting for inference'
                    }
                    print("WARNING: Buffer empty!")
                else:
                    if self.use_async_rtc:
                        response = {"chunk": action, "metadata": metadata}
                        response["metadata"]["send_time"] = time.time()
                    else:
                        response = action
                        # Track stale actions
                        if action.get('is_stale', False):
                            with self.stats_lock:
                                self.stats['buffer_wraps'] += 1
                            
                            # Only print occasionally to avoid spam
                            if self.stats['buffer_wraps'] % 10 == 1:
                                print(f"WARNING: Serving stale actions (total: {self.stats['buffer_wraps']}). "
                                    f"Inference too slow or chunk too small.")
                        
                        # Log buffer status occasionally
                        remaining = action.get('buffer_remaining', 0)
                        if remaining == 2:
                            print(f"Buffer running low: {remaining} actions remaining")
                
                # Send action to robot
                self.action_socket.send_pyobj(response)
                
                with self.stats_lock:
                    self.stats['actions_served'] += 1
                
                # Periodic stats
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
        """
        Listen for control commands from robot.
        Handles commands like clearing the action queue.
        """
        print("Control thread started - listening for commands", flush=True)
        
        while not self.stop_event.is_set():
            try:
                # Wait for control command from robot (blocking)
                command = self.control_socket.recv_pyobj()
                print(f"Received control command: {command}", flush=True)
                
                if command.get('command') == 'clear_queue':
                    # Clear the action buffer
                    self.action_buffer.clear()
                    
                    # Clear the policy's internal action queue
                    try:
                        self.model.reset_action_queue()
                        print("Policy action queue reset")
                    except Exception as e:
                        print(f"Warning: Could not reset policy queue: {e}")
                    
                    # Send acknowledgment
                    response = {'status': 'ok', 'message': 'Action queue cleared'}
                else:
                    response = {'status': 'error', 'message': f'Unknown command: {command.get("command")}'}
                
                self.control_socket.send_pyobj(response)
                
            except Exception as e:
                print(f"Control thread error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        
        print("Control thread stopped")

    
    def preprocess_observation(self, observation):
        """
        Convert observation dict to model input format.
        
        Args:
            observation: Dict with 'left_ee_pose', 'right_ee_pose', 'timestamp'
        Returns:
            Preprocessed tensor ready for model input
        """
        # left_pose = observation['left_ee_pose']
        # right_pose = observation['right_ee_pose']
        
        # obs_array = np.concatenate([left_pose, right_pose])
        # obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0) # bs=1
        # obs_tensor = obs_tensor.to(self.device)
        
        # return obs_tensor

        return observation  # Assuming model handles raw observation dict
    
    def postprocess_action_chunk(self, model_output, observation=None):
        """
        Convert model output to list of action dicts.
        
        Args:
            model_output: Raw output from policy model
                         Shape: [1, chunk_size, action_dim] or [chunk_size, action_dim]
            
        Returns:
            List of action dicts, one per timestep
        """
        if isinstance(model_output, torch.Tensor):
            output = model_output.detach().cpu().numpy()
        else:
            output = model_output
        
        if output.ndim == 3:
            output = output.squeeze(0)
        
        action_chunk = []

        if self.model.is_delta_action:
            for t in range(output.shape[0]):
                action_t = output[t]
                
                action = {
                    'left_delta_pose': action_t[0:7],
                    'right_delta_pose': action_t[7:14],
                    'left_gripper': float(action_t[14]),
                    'right_gripper': float(action_t[15]),
                    'timestamp': time.time(),
                    'chunk_index': t
                }

                if observation is not None:
                    action['obs_timestamp'] = observation.get('timestamp', None)

                action_chunk.append(action)
        else:
            for t in range(output.shape[0]):
                action_t = output[t]
                
                action = {
                    'left_ee_pose': action_t[0:7],
                    'right_ee_pose': action_t[7:14],
                    'left_gripper': float(action_t[14]),
                    'right_gripper': float(action_t[15]),
                    'timestamp': time.time(),
                    'chunk_index': t
                }

                if observation is not None:
                    action['obs_timestamp'] = observation.get('timestamp', None)

                action_chunk.append(action)
        
        return action_chunk
    
    def _run_inference_chunk(self, observation):
        """Run model inference to generate a chunk of actions."""
        model_input = self.preprocess_observation(observation)

        # Use async RTC if enabled
        if self.use_async_rtc:
            with torch.no_grad():
                # Call async RTC method which returns (actions_to_execute, metadata)
                # actions_to_execute has shape (chunk_size, action_dim) in quaternion format
                model_output, metadata = self.model.predict_action_chunk_async(
                    model_input,
                    transform_to_quat=True
                )

            # Log RTC metadata
            print(f"DEBUG: Async RTC inference - chunk_idx={metadata['chunk_idx']}, "
                  f"prefix_length={metadata['prefix_length']} (d), "
                  f"execution_horizon={metadata['execution_horizon']} (s)", flush=True)

            # predict_action_chunk_async already returns postprocessed quaternion actions
            # Just need to convert to action_chunk format for the buffer
            action_chunk = self.postprocess_action_chunk(model_output, observation)
        else:
            # Standard synchronous inference
            with torch.no_grad():
                model_output = self.model.forward(model_input)

            action_chunk = self.postprocess_action_chunk(model_output, observation)

        print("DEBUG: Generated action in InferenceServer")
        return (action_chunk, metadata) if self.use_async_rtc else (action_chunk, None)
    
    # def _run_inference_single(self, observation):
    #     """Run model inference to generate a chunk of actions."""
    #     model_input = self.preprocess_observation(observation)

    #     #Policy class wants a obs dict 
    #     with torch.no_grad():
    #         model_output = self.model.forward(model_input)
        
    #     action_chunk = self.postprocess_action_chunk(model_output, observation)
    #     print("DEBUG: Generated action in InferenceServer")
    #     return action_chunk
    
    def _print_stats(self):
        """Print current statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        buffer_status = self.action_buffer.get_status()
        
        print(f"\n{'='*60}")
        print(f"Stats after {stats['actions_served']} actions served:")
        print(f"  Observations received: {stats['observations_received']}")
        print(f"  Inferences completed: {stats['inferences_completed']}")
        print(f"  Actions served: {stats['actions_served']}")
        
        if stats['inferences_completed'] > 0:
            avg_inf_time = stats['total_inference_time'] / stats['inferences_completed']
            print(f"  Avg inference time: {avg_inf_time*1000:.1f}ms")
            print(f"  Inference frequency: {1.0/avg_inf_time:.1f} Hz")
        
        print(f"  Stale actions served: {stats['buffer_wraps']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Buffer status:")
        print(f"    - Actions remaining: {buffer_status['actions_remaining']}/{buffer_status['chunk_size']}")
        print(f"    - Buffer updates: {buffer_status['update_count']}")
        print(f"    - Total pops: {buffer_status['total_pops']}")
        if buffer_status['age'] is not None:
            print(f"    - Age since last update: {buffer_status['age']*1000:.1f}ms")
        print(f"{'='*60}\n")
    
    def run(self):
        """Start both inference and action serving threads."""
        print("\nStarting asynchronous inference server...")
        print("  - Inference thread: Continuously runs on latest observations")
        print("  - Action thread: Serves actions at robot's request rate\n")
        print("  - Control thread: Handles control commands from robot\n")
        
        self.inference_thread.start()
        self.action_thread.start()
        self.control_thread.start()
        
        try:
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown."""
        self.stop_event.set()
        
        self.inference_thread.join(timeout=2.0)
        self.action_thread.join(timeout=2.0)
        self.control_thread.join(timeout=2.0)
        
        print("\n" + "="*60)
        print("Final Statistics:")
        self._print_stats()
        
        self.obs_socket.close()
        self.action_socket.close()
        self.control_socket.close()
        self.context.term()
        print("Server stopped")


def load_policy_model(checkpoint_path, chunk_size, args, device='cuda'):
    """
    Load your trained policy model with async RTC support.

    Args:
        checkpoint_path: Path to model checkpoint
        chunk_size: Number of actions to predict per inference (full chunk size)
        args: Arguments containing inference delay and control frequency
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model
    """
    # Load policy with async RTC parameters
    policy = Pi05InferencePolicy(
        is_delta_action=False,
        checkpoint_path=checkpoint_path,
        device="cuda",
        primary_camera='cam_high',
        control_freq=args.hz,
        inference_delay=args.inference_delay
    )

    return policy


def main():
    parser = argparse.ArgumentParser(
        description='Asynchronous HPC Inference Server with Continuous Inference and RTC'
    )
    parser.add_argument('--obs-port', type=int, default=5555,
                      help='Port for receiving observations')
    parser.add_argument('--action-port', type=int, default=5556,
                      help='Port for serving actions')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to run inference on')
    parser.add_argument("--pred_horizon", default=16, type=int, help="Action prediction horizon (full chunk size)")
    parser.add_argument("--inference-delay", default=3, type=int,
                      help="Inference delay in action steps (d) - actions executed during inference + network delay")
    parser.add_argument("--exp_weight", default=0.01, type=float, help="Exponential weight for ensemble")
    parser.add_argument("--temp_ensemble", action="store_true", help="Use temporal ensembling")
    parser.add_argument("--hz", default=30, type=float, help="Control frequency")
    parser.add_argument("--use-async-rtc", action="store_true",
                      help="Use async RTC inference (default: False)")
    args = parser.parse_args()
    args.period = 1.0 / args.hz
    
    print(f"Loading policy model ...", flush=True)
    model = load_policy_model(args.checkpoint, chunk_size=args.pred_horizon, args=args, device=args.device)

    # Action buffer size should be the full prediction horizon (H)
    # We send the entire chunk to robot, which manages buffer updates
    buffer_chunk_size = args.pred_horizon

    print(f"Server configuration:", flush=True)
    print(f"  Async RTC: {args.use_async_rtc}", flush=True)
    print(f"  Full prediction horizon (H): {args.pred_horizon}", flush=True)
    print(f"  Inference delay (d): {args.inference_delay}", flush=True)
    print(f"  Action buffer size: {buffer_chunk_size}", flush=True)
    print(f"  Control frequency: {args.hz} Hz", flush=True)

    server = InferenceServer(
        model,
        obs_port=args.obs_port,
        action_port=args.action_port,
        device=args.device,
        chunk_size=buffer_chunk_size,
        use_async_rtc=args.use_async_rtc
    )
    server.run()


if __name__ == "__main__":
    main()