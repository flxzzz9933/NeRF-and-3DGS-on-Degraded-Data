import json, sys, os
import numpy as np
from pathlib import Path

in_json = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir, exist_ok=True)

with open(in_json, "r") as f:
    data = json.load(f)

frames = data["frames"]
camera_angle_x = data.get("camera_angle_x", 0.691)

# Try reading resolution from transforms.json (if exists)
W = data.get("w", 1920)
H = data.get("h", 1080)

fx = (W / 2) / np.tan(camera_angle_x / 2)
fy = fx
cx = W / 2
cy = H / 2

### --------- cameras.txt ---------
with open(os.path.join(out_dir, "cameras.txt"), "w") as f_cam:
    f_cam.write("# Camera list\n")
    f_cam.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
    f_cam.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

### --------- images.txt ---------
with open(os.path.join(out_dir, "images.txt"), "w") as f_img:
    f_img.write("# Image list\n")
    f_img.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")

    for i, fr in enumerate(frames):
        m = np.array(fr["transform_matrix"])
        R = m[:3,:3]
        t = m[:3,3]

        # Convert OpenGL -> COLMAP
        R_col = R.T
        t_col = -R_col @ t

        # Quaternion
        qw = np.sqrt(1 + R_col[0,0] + R_col[1,1] + R_col[2,2]) / 2
        qx = (R_col[2,1] - R_col[1,2]) / (4 * qw)
        qy = (R_col[0,2] - R_col[2,0]) / (4 * qw)
        qz = (R_col[1,0] - R_col[0,1]) / (4 * qw)

        # Extract filename only
        fname = Path(fr["file_path"]).name

        f_img.write(f"{i+1} {qw} {qx} {qy} {qz} {t_col[0]} {t_col[1]} {t_col[2]} 1 {fname}\n")
        f_img.write("\n")

### --------- points3D.txt ---------
with open(os.path.join(out_dir, "points3D.txt"), "w") as f_pts:
    f_pts.write("# Empty points3D\n")

print("Converted OK →", out_dir)
