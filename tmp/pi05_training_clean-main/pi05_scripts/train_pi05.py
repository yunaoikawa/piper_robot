#!/usr/bin/env python3
"""
Training script for fine-tuning Pi0.5 model on LIBERO-90 dataset.

This script uses LeRobot's implementation of the Pi0.5 (π₀.₅) model
from Physical Intelligence and fine-tunes it on the LIBERO-90 benchmark.

Usage:
    python train_pi05_libero.py --config configs/pi05_libero90_config.yaml

    Or use LeRobot's CLI directly:
    lerobot-train \\
        --policy.type=pi05 \\
        --policy.pretrained_model_name_or_path=lerobot/pi05_libero_base \\
        --dataset.repo_id=HuggingFaceVLA/libero \\
        --env.type=libero \\
        --env.task=libero_90 \\
        --output_dir=./outputs/pi05_libero90 \\
        --policy.compile_model=true \\
        --policy.gradient_checkpointing=true \\
        --training.num_epochs=100 \\
        --training.batch_size=32 \\
        --training.learning_rate=1e-4 \\
        --wandb.enable=true \\
        --wandb.project=pi05-libero90-finetuning
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("✓ LeRobot imports successful")
except ImportError as e:
    print(f"Error importing LeRobot: {e}")
    print("\nPlease install LeRobot with:")
    print('  pip install "lerobot[pi,libero]@git+https://github.com/huggingface/lerobot.git"')
    sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment():
    """Setup environment variables and check dependencies."""
    # Set MuJoCo rendering backend for headless servers
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Set MUJOCO_GL=egl for headless rendering")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available, training will use CPU (slow)")

    return torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Pi0.5 on LIBERO-90')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pi05_libero90_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory to save outputs and checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with reduced dataset size'
    )

    args = parser.parse_args()

    # Setup environment
    # print("=" * 80)
    # print("Pi0.5 Fine-tuning on LIBERO-90")
    # print("=" * 80)
    # print()

    has_cuda = setup_environment()

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"✓ Loaded config from {args.config}")
    else:
        print(f"⚠ Config file not found: {args.config}")
        print("  Using default configuration")
        config = {
            'policy': {
                'type': 'pi05',
                'pretrained_model_name_or_path': 'lerobot/pi05_libero_base',
                'compile_model': True,
                'gradient_checkpointing': True,
                'batch_size': 32,
            },
            'dataset': {
                'repo_id': 'HuggingFaceVLA/libero',
            },
            'env': {
                'type': 'libero',
                'task': 'libero_90',
            },
            'training': {
                # 'num_epochs': 100,
                'steps': 100000,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'warmup_steps': 1000,
                'gradient_accumulation_steps': 1,
                'learning_rate': 1e-4,
            }
        }

    # Override with command line arguments
    # config['output_dir'] = args.output_dir
    # config['wandb'] = {'enable': args.wandb, 'project': 'pi05-libero90-finetuning'}
    if args.debug:
        config['training']['num_epochs'] = 2
        config['training']['batch_size'] = 4
        print("⚠ Running in DEBUG mode with reduced epochs and batch size")

    print()
    print("Configuration:")
    print("-" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 80)
    print()

    is_distributed = config.get('distributed', None)
    if is_distributed == 'true' or is_distributed is True:
        # print("⚠ Distributed training is enabled. Please run with 'lerobot-train' CLI for best results.")
        # sys.exit(1)
        cmd_parts = [
            'accelerate', 'launch',
            '--multi_gpu',
            f'--num_processes={config.get("num_gpus", 2)}',
            '$(which lerobot-train)',
            f'--steps={config["training"]["steps"]}',
            f'--policy.type={config["policy"]["type"]}',
            f'--policy.pretrained_path={config["policy"]["pretrained_model_name_or_path"]}',
            f'--policy.repo_id="N/A"',
            f'--dataset.repo_id={config["dataset"]["repo_id"]}',
            f'--output_dir={config["output_dir"]}',
            f'--batch_size={config["training"]["batch_size"]}',
            f'--optimizer.lr={config["training"]["learning_rate"]}',
            f'--eval.batch_size=1',
            f'--eval.n_episodes=2',
            f'--save_freq=500',
            f'--eval_freq=2000', # eval is not working without a sim env
        ]

        # Add optional flags
        if config["dataset"].get("root", None):
            cmd_parts.append(f'--dataset.root={config["dataset"]["root"]}')
            # cmd_parts.append(f'--dataset.local_files_only={str(config["dataset"].get("local_files_only", False)).lower()}')
        if config["dataset"].get("val_root", None):
            cmd_parts.append(f'--dataset.val_root={config["dataset"]["val_root"]}')
            cmd_parts.append(f'--dataset.val_repo_id={config["dataset"].get("val_repo_id", "")}')
            if config["training"].get("val_freq", None):
                cmd_parts.append(f'--val_freq={config["training"]["val_freq"]}')
            else:
                cmd_parts.append(f'--val_freq=500')
        if config.get("env", {}).get("type", None):
            cmd_parts.append(f'--env.type={config["env"]["type"]}')
        if config.get("env", {}).get("task", None):
            cmd_parts.append(f'--env.task={config["env"]["task"]}')
        if config['policy'].get('compile_model', False):
            cmd_parts.append('--policy.compile_model=true')
            if config['policy'].get('compile_mode', None):
                cmd_parts.append(f'--policy.compile_mode={config["policy"]["compile_mode"]}')
        if config['policy'].get('gradient_checkpointing', False):
            cmd_parts.append('--policy.gradient_checkpointing=true')
        if config['policy'].get('chunk_size', None):
            cmd_parts.append(f'--policy.chunk_size={config["policy"]["chunk_size"]}')
        if config['policy'].get('n_action_steps', None):
            cmd_parts.append(f'--policy.n_action_steps={config["policy"]["n_action_steps"]}')
        if config.get('wandb', {}).get('enable', False):
            cmd_parts.append('--wandb.enable=true')
            cmd_parts.append(f'--wandb.project={config["wandb"]["project"]}')
            cmd_parts.append(f'--wandb.entity={config["wandb"].get("entity", "")}')
            run_name = config["wandb"].get("name", "")
            run_name += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cmd_parts.append(f'--wandb.run_id={run_name}')
            job_name = config.get("job_name", "lerobot_job")
            job_name += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cmd_parts.append(f'--job_name={job_name}')
            # cmd_parts.append(f'--wandb.name={config["wandb"].get("name", "")}')
        if args.resume:
            cmd_parts.append(f'--resume={args.resume}')
        # add depth
        if config["policy"].get("depth_max_value", None):
            cmd_parts.append(f'--policy.depth_max_value={config["policy"]["depth_max_value"]}')
        if config["policy"].get("image_features", None):
            cmd_parts.append(f'--policy.image_feature_keys={json.dumps(config["policy"]["image_features"], separators=(",", ":"))}')

    else:
        print("✓ Single-node training")

        # Build LeRobot training command
        cmd_parts = [
            'lerobot-train',
            f'--steps={config["training"]["steps"]}',
            f'--policy.type={config["policy"]["type"]}',
            f'--policy.pretrained_path={config["policy"]["pretrained_model_name_or_path"]}',
            f'--policy.repo_id="N/A"',
            f'--dataset.repo_id={config["dataset"]["repo_id"]}',
            f'--output_dir={config["output_dir"]}',
            f'--batch_size={config["training"]["batch_size"]}',
            f'--optimizer.lr={config["training"]["learning_rate"]}',
            f'--policy.device={config.get("device", None)}',
            f'--eval.batch_size=1',
            f'--eval.n_episodes=2',
            f'--save_freq=500',
            f'--eval_freq=2000',  # eval is not working without a sim env
        ]
            

        # Add optional flags
        if config["dataset"].get("root", None):
            cmd_parts.append(f'--dataset.root={config["dataset"]["root"]}')
            # cmd_parts.insert(0, )
        if config["dataset"].get("val_root", None):
            cmd_parts.append(f'--dataset.val_root={config["dataset"]["val_root"]}')
            cmd_parts.append(f'--dataset.val_repo_id={config["dataset"].get("val_repo_id", "")}')
            if config["training"].get("val_freq", None):
                cmd_parts.append(f'--val_freq={config["training"]["val_freq"]}')
            else:
                cmd_parts.append(f'--val_freq=500')
        if config.get("env", {}).get("type", None):
            cmd_parts.append(f'--env.type={config["env"]["type"]}')
        if config.get("env", {}).get("task", None):
            cmd_parts.append(f'--env.task={config["env"]["task"]}')
        if config['policy'].get('compile_model', False):
            cmd_parts.append('--policy.compile_model=true')
            if config['policy'].get('compile_mode', None):
                cmd_parts.append(f'--policy.compile_mode={config["policy"]["compile_mode"]}')
        if config['policy'].get('gradient_checkpointing', False):
            cmd_parts.append('--policy.gradient_checkpointing=true')
        if config['policy'].get('chunk_size', None):
            cmd_parts.append(f'--policy.chunk_size={config["policy"]["chunk_size"]}')
        if config['policy'].get('n_action_steps', None):
            cmd_parts.append(f'--policy.n_action_steps={config["policy"]["n_action_steps"]}')
        if config.get('wandb', {}).get('enable', False):
            cmd_parts.append('--wandb.enable=true')
            cmd_parts.append(f'--wandb.project={config["wandb"]["project"]}')
            cmd_parts.append(f'--wandb.entity={config["wandb"].get("entity", "")}')
            run_name = config["wandb"].get("name", "")
            run_name += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cmd_parts.append(f'--wandb.run_id={run_name}')
            job_name = config.get("job_name", "lerobot_job")
            job_name += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cmd_parts.append(f'--job_name={job_name}')
        if args.resume:
            cmd_parts.append(f'--resume={args.resume}')
        # add depth
        if config["policy"].get("depth_max_value", None):
            cmd_parts.append(f'--policy.depth_max_value={config["policy"]["depth_max_value"]}')
        if config["policy"].get("image_features", None):
            cmd_parts.append(f'--policy.image_feature_keys={json.dumps(config["policy"]["image_features"], separators=(",", ":"))}')

    # Print and execute command
    cmd = ' \\\n    '.join(cmd_parts)
    print("Executing LeRobot training command:")
    print("-" * 80)
    print(cmd)
    print("-" * 80)
    print()

    # Execute training
    import subprocess
    try:
        env = os.environ.copy()
        env['NCCL_TIMEOUT'] = '3600'
        env['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '3600'
        subprocess.run(['bash', '-c', ' '.join(cmd_parts)], check=True, env=env)
        print()
        print("=" * 80)
        print("✓ Training completed successfully!")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print(f"✗ Training failed with error code {e.returncode}")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
