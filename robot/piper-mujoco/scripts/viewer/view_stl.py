import argparse
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 引数
parser = argparse.ArgumentParser(description="Render STL to PNG without Open3D")
parser.add_argument("stl_path", type=str, help="Path to input STL file")
parser.add_argument("--output", type=str, default="output.png", help="Output PNG file")
args = parser.parse_args()

# STL読み込み
mesh = trimesh.load(args.stl_path)

# matplotlibで描画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# メッシュの三角形を描画
ax.plot_trisurf(
    mesh.vertices[:, 0],
    mesh.vertices[:, 1],
    mesh.vertices[:, 2],
    triangles=mesh.faces,
    linewidth=0.2,
    antialiased=True
)

# 見た目調整
ax.set_axis_off()
ax.view_init(elev=20, azim=30)

# 保存
plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
plt.close()