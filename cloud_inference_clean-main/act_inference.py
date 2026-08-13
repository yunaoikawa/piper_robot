"""
ACT (Action Chunking Transformer) Inference Class for LeRobot

Drop-in analogue of Pi05InferencePolicy for ACT checkpoints. Differences vs. Pi0.5:
    - No RTC (real-time chunking) prefix conditioning
    - No torch.compile warmup
    - No language ("task") conditioning passed to the model
    - Standard chunked inference via ACTPolicy.predict_action_chunk()

Assumes the same 20D r6 action layout as the Pi0.5 setup
(left_pos(3), left_r6(6), left_grip(1), right_pos(3), right_r6(6), right_grip(1)),
which is converted to a 16D quaternion layout for the robot.

Example:
    from act_inference import ACTInferencePolicy

    policy = ACTInferencePolicy(checkpoint_path="/path/to/ckpt", device="cuda")
    policy.warmup()

    observation = {
        "images": {"cam_high": img_np, "cam_left": img_np, "cam_right": img_np},
        "qpos": np.array(...),  # (20,)
    }
    action = policy.forward(observation)   # (16,) quaternion format
"""

import json
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Ensure local imports resolve
_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
except (ImportError, ModuleNotFoundError):
    try:
        from policies.act.configuration_act import ACTConfig
        from policies.act.modeling_act import ACTPolicy
        from processor.pipeline import PolicyProcessorPipeline
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import LeRobot ACT modules. Make sure lerobot is installed. "
            f"Error: {e}"
        )


class ACTInferencePolicy:
    """
    Inference wrapper for an ACT policy checkpoint.

    The checkpoint directory should contain:
        - config.json
        - model.safetensors
        - policy_preprocessor.json (+ normalizer .safetensors)
        - policy_postprocessor.json (+ unnormalizer .safetensors)
        - train_config.json (optional)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        verbose: bool = True,
        primary_camera: str | None = None,
        is_delta_action: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.verbose = verbose
        self.primary_camera = primary_camera
        self.is_delta_action = is_delta_action

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_path}")

        if self.verbose:
            print(f"Loading ACT model from {self.checkpoint_path}")

        self.policy = self._load_policy()
        self.config = self.policy.config
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor = self._load_processor("policy_preprocessor.json", "preprocessor")
        self.postprocessor = self._load_processor("policy_postprocessor.json", "postprocessor")
        self.train_config = self._load_train_config()

        self.reset()

        if self.verbose:
            print("✓ ACT model loaded successfully")
            self._print_model_info()

    # ── Loading ────────────────────────────────────────────────────────────────
    def _load_policy(self) -> ACTPolicy:
        if self.verbose:
            print(f"  Loading policy from {self.checkpoint_path}")
        return ACTPolicy.from_pretrained(str(self.checkpoint_path))

    def _load_processor(self, config_filename: str, kind: str) -> PolicyProcessorPipeline:
        if self.verbose:
            print(f"  Loading {kind} ({config_filename})")
        return PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint_path), config_filename=config_filename
        )

    def _load_train_config(self) -> dict | None:
        path = self.checkpoint_path / "train_config.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _print_model_info(self):
        print("\nModel Information:")
        print(f"  Config:           {self.config.__class__.__name__}")
        print(f"  Image keys:       {self.expected_image_keys}")
        print(f"  State dim:        {self.expected_state_dim}")
        print(f"  Action dim:       {self.action_dim}")
        print(f"  Chunk size:       {self.chunk_size}")
        print(f"  n_action_steps:   {getattr(self.config, 'n_action_steps', '?')}")
        print(f"  Device:           {self.device}")

    # ── State ──────────────────────────────────────────────────────────────────
    def reset(self):
        """Reset the policy's internal action queue."""
        self.policy.reset()

    def reset_action_queue(self):
        """Clear ACT's internal FIFO action queue."""
        if hasattr(self.policy, "_action_queue") and isinstance(self.policy._action_queue, deque):
            self.policy._action_queue.clear()
        else:
            self.policy._action_queue = deque(maxlen=self.chunk_size)

    def warmup(self):
        """Run a single forward pass with dummy inputs to warm CUDA kernels/caches."""
        print("\n" + "=" * 60)
        print("Running ACT warmup pass (CUDA cache, cuDNN autotuning)...")
        print("=" * 60)

        dummy_obs: dict[str, Any] = {}
        if self.expected_state_dim > 0:
            dummy_obs["observation.state"] = torch.randn(self.expected_state_dim, device=self.device)

        if hasattr(self.config, "input_features"):
            for k in self.expected_image_keys:
                shape = self.config.input_features[k].shape
                dummy_obs[k] = torch.randn(1, *shape, device=self.device)
            if "observation.depth" in self.config.input_features:
                shape = self.config.input_features["observation.depth"].shape
                dummy_obs["observation.depth"] = torch.randn(1, *shape, device=self.device)

        try:
            with torch.inference_mode():
                _ = self.predict(dummy_obs)
            print("✓ Warmup complete")
        except Exception as e:
            print(f"✗ Warmup failed: {e}")
            raise

    # ── Rotation conversions (identical to Pi0.5 path) ─────────────────────────
    def r6_to_quat(self, delta_r6: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply a delta r6 rotation on top of the current r6 rotation; return wxyz quat."""
        if state.ndim == 1:
            state = state[None]
        if delta_r6.ndim == 1:
            delta_r6 = delta_r6[None]

        r6 = delta_r6 + state[..., 3:9]
        mat = self._r6_to_matrix(r6)
        prev_mat = self._r6_to_matrix(state[..., 3:9])
        relative_mat = mat @ np.transpose(prev_mat, (0, 2, 1))  # world frame
        quat = R.from_matrix(relative_mat).as_quat(scalar_first=True)
        return quat.squeeze() if quat.shape[0] == 1 else quat

    def r6_absolute_to_quat(self, r6: np.ndarray) -> np.ndarray:
        if r6.ndim == 1:
            r6 = r6[None]
        mat = self._r6_to_matrix(r6)
        quat = R.from_matrix(mat).as_quat(scalar_first=True)
        return quat.squeeze() if quat.shape[0] == 1 else quat

    @staticmethod
    def _r6_to_matrix(r6: np.ndarray) -> np.ndarray:
        r1, r2 = r6[..., :3], r6[..., 3:6]
        b1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
        b2 = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)

    # ── Observation/action transforms ──────────────────────────────────────────
    def _get_expected_image_keys(self) -> list[str]:
        if hasattr(self.config, "input_features"):
            return [k for k in self.config.input_features.keys() if k.startswith("observation.image")]
        return []

    def _transform_observation_to_lerobot_format(
        self, obs: dict[str, Any], image_size: tuple[int, int] = (224, 224)
    ) -> dict[str, Any]:
        lerobot_obs: dict[str, Any] = {}

        if "qpos" in obs:
            qpos = obs["qpos"]
            lerobot_obs["observation.state"] = (
                torch.from_numpy(qpos).float() if isinstance(qpos, np.ndarray) else qpos
            )

        def process_image(img):
            if isinstance(img, np.ndarray):
                if img.shape[:2] != (480, 640):
                    # Inference nodes are headless and need not provide libGL.
                    # Pillow keeps this resize independent of GUI OpenCV wheels.
                    img = np.asarray(Image.fromarray(img).resize((640, 480), Image.Resampling.BOX))
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                return torch.from_numpy(img).float().permute(2, 0, 1)
            return img

        if "images" in obs:
            expected = self._get_expected_image_keys()
            if expected and len(expected) == 1:
                if self.primary_camera and self.primary_camera in obs["images"]:
                    lerobot_obs[expected[0]] = process_image(obs["images"][self.primary_camera])
                else:
                    cam_name, img = next(iter(obs["images"].items()))
                    lerobot_obs[expected[0]] = process_image(img)
                    if self.primary_camera:
                        warnings.warn(
                            f"Primary camera '{self.primary_camera}' not found; using '{cam_name}'."
                        )
            else:
                for cam_name, img in obs["images"].items():
                    lerobot_obs[f"observation.images.{cam_name}"] = process_image(img)

        if "depth" in obs:
            depth = obs["depth"]
            if isinstance(depth, np.ndarray):
                if depth.ndim == 2:
                    depth_t = torch.from_numpy(depth).float().unsqueeze(0)
                elif depth.ndim == 3 and depth.shape[2] == 1:
                    depth_t = torch.from_numpy(depth).float().permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected depth shape: {depth.shape}")
                lerobot_obs["observation.depth"] = depth_t
            else:
                lerobot_obs["observation.depth"] = depth

        # ACT does not consume "task", but pass through if downstream processors expect it.
        if "task" in obs:
            lerobot_obs["task"] = obs["task"]

        return lerobot_obs

    def _transform_rel_transform_to_quat_format(self, action: np.ndarray) -> np.ndarray:
        """20D absolute r6 action → 16D quaternion action."""
        single_action = action.ndim == 1
        if single_action:
            action = action[None, :]

        B = action.shape[0]
        out = np.zeros((B, 16), dtype=np.float32)

        out[:, 0:4] = self.r6_absolute_to_quat(action[:, 3:9])
        out[:, 4:7] = action[:, 0:3]
        out[:, 7:11] = self.r6_absolute_to_quat(action[:, 13:19])
        out[:, 11:14] = action[:, 10:13]
        out[:, 14] = action[:, 9]
        out[:, 15] = action[:, 19]

        return out.squeeze() if single_action else out

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(
        self,
        observation: dict[str, Any],
        return_dict: bool = False,
        transform_to_quat: bool = False,
    ) -> np.ndarray | dict:
        """Single-step action prediction (uses ACT's internal queue)."""
        is_raw = "images" in observation and "qpos" in observation
        if is_raw:
            observation = self._transform_observation_to_lerobot_format(observation)

        processed = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(processed)
        action = self.postprocessor({"action": action})

        action = self._to_numpy(action, return_dict=return_dict)
        if transform_to_quat and isinstance(action, np.ndarray):
            action = self._transform_rel_transform_to_quat_format(action)
        return action

    def predict_action_chunk(
        self,
        observation: dict[str, Any],
        return_dict: bool = False,
        transform_to_quat: bool = False,
    ) -> np.ndarray | dict:
        """Predict a full action chunk (chunk_size actions)."""
        is_raw = "images" in observation and "qpos" in observation
        if is_raw:
            observation = self._transform_observation_to_lerobot_format(observation)

        processed = self.preprocessor(observation)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(processed)
        chunk = self.postprocessor({"action": chunk})

        chunk = self._to_numpy(chunk, return_dict=return_dict)
        if transform_to_quat and isinstance(chunk, np.ndarray):
            chunk = self._transform_rel_transform_to_quat_format(chunk)
        return chunk

    def forward(self, obs: dict[str, Any]) -> np.ndarray:
        """Sample-policy interface: raw observation → 16D quaternion action."""
        lerobot_obs = self._transform_observation_to_lerobot_format(obs)
        processed = self.preprocessor(lerobot_obs)

        with torch.inference_mode():
            action = self.policy.select_action(processed)
        action = self.postprocessor({"action": action})
        action = self._to_numpy(action)

        return self._transform_rel_transform_to_quat_format(action)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_numpy(action, return_dict: bool = False):
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy()
        if isinstance(action, dict):
            if return_dict:
                return {
                    k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in action.items()
                }
            if "action" in action:
                v = action["action"]
                return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        return action

    # ── Properties ─────────────────────────────────────────────────────────────
    @property
    def expected_image_keys(self) -> list[str]:
        return self._get_expected_image_keys()

    @property
    def expected_state_dim(self) -> int:
        if not hasattr(self.config, "input_features"):
            return 0
        meta = self.config.input_features.get("observation.state")
        if meta is None:
            return 0
        s = meta.shape
        return s[0] if isinstance(s, (list, tuple)) else s

    @property
    def action_dim(self) -> int:
        if hasattr(self.config, "output_features") and "action" in self.config.output_features:
            af = self.config.output_features["action"]
            shape = af.shape if hasattr(af, "shape") else af["shape"]
            return shape[0]
        return 0

    @property
    def chunk_size(self) -> int:
        return getattr(self.config, "chunk_size", 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ACT Inference smoke-test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading ACT Policy")
    print("=" * 60)
    policy = ACTInferencePolicy(args.checkpoint, device=args.device)

    print("\nProperties:")
    print(f"  Image keys: {policy.expected_image_keys}")
    print(f"  State dim:  {policy.expected_state_dim}")
    print(f"  Action dim: {policy.action_dim}")
    print(f"  Chunk size: {policy.chunk_size}")

    policy.warmup()

    if args.test:
        dummy: dict[str, Any] = {}
        if policy.expected_state_dim > 0:
            dummy["observation.state"] = torch.randn(policy.expected_state_dim)
        for k in policy.expected_image_keys:
            dummy[k] = torch.randn(3, 224, 224)

        action = policy.predict(dummy)
        print(f"\n✓ single action shape: {action.shape}")
        chunk = policy.predict_action_chunk(dummy)
        print(f"✓ chunk shape:         {chunk.shape}")


if __name__ == "__main__":
    main()
