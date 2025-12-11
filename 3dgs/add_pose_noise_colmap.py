import sys, os
import numpy as np

images_path = sys.argv[1]
rot_std_deg = float(sys.argv[2])
trans_std = float(sys.argv[3])

def q_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ])
    return R

def R_to_q(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return qw, qx, qy, qz

def sample_rot_noise(rot_std_deg):
    if rot_std_deg <= 0:
        return np.eye(3)
    std_rad = np.deg2rad(rot_std_deg)
    v = np.random.normal(0.0, std_rad, 3)
    angle = np.linalg.norm(v)
    if angle < 1e-8:
        return np.eye(3)
    axis = v / angle
    x, y, z = axis
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)
    return R

lines = open(images_path, "r").readlines()
out = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    # 注释行 / 空行原样保留
    if stripped.startswith("#") or stripped == "":
        out.append(line)
        i += 1
        continue
    parts = stripped.split()
    # 非标准行直接保留
    if len(parts) < 10:
        out.append(line)
        i += 1
        continue

    img_id = int(parts[0])
    qw, qx, qy, qz = map(float, parts[1:5])
    tx, ty, tz = map(float, parts[5:8])
    cam_id = parts[8]
    name = parts[9]

    R = q_to_R(qw, qx, qy, qz)
    R_noise = sample_rot_noise(rot_std_deg)
    R_new = R_noise @ R
    qw2, qx2, qy2, qz2 = R_to_q(R_new)

    t = np.array([tx, ty, tz], dtype=float)
    if trans_std > 0:
        t += np.random.normal(0.0, trans_std, 3)

    new_line = f"{img_id} {qw2} {qx2} {qy2} {qz2} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n"
    out.append(new_line)

    # 紧跟着的一行是 POINTS2D，可直接原样保留
    if i + 1 < len(lines):
        out.append(lines[i+1])
        i += 2
    else:
        i += 1

with open(images_path, "w") as f:
    f.writelines(out)

print("Updated poses with noise:", images_path)
