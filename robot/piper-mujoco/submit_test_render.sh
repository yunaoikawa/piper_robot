#!/bin/bash
#SBATCH --job-name=test_render         # ジョブ名
#SBATCH --output=logs/test_render_%j.out  # 標準出力ログ（%jはジョブID）
#SBATCH --error=logs/test_render_%j.err   # 標準エラーログ
#SBATCH --partition=gpu                # パーティション名（環境に合わせて変更）
#SBATCH --gres=gpu:1                   # GPU 1枚を要求
#SBATCH --cpus-per-task=4             # CPUコア数
#SBATCH --mem=16G                      # メモリ

# 作業ディレクトリに移動
cd /home/rshimayoshi/mujoco

# 仮想環境を有効化
source pyenv-mujoco/bin/activate

# スクリプトを実行
python scripts/exp/test_render.py