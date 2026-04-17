#!/bin/bash

# 対象ディレクトリ
DIR="/usr/tmp/robot/cone-e-description/meshes/piper"

# Pythonスクリプト名
SCRIPT="scripts/stl_viewer/stl_view.py"

# ループ処理
for stl in "$DIR"/*.stl; do
    # ファイル名（拡張子なし）を取得
    filename=$(basename "$stl" .stl)

    # 出力ファイル名
    output="/home/rshimayoshi/mujoco/image/mesh_image/${filename}.png"

    echo "Processing: $stl -> $output"

    python "$SCRIPT" "$stl" --output "$output"
done

echo "Done."