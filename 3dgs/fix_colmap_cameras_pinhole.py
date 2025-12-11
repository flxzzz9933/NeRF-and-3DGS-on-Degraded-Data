import sys, os
import numpy as np

path = sys.argv[1]

lines = open(path, "r").read().splitlines()
out = []

for line in lines:
    if line.strip().startswith("#") or line.strip() == "":
        out.append(line + "\n")
        continue
    parts = line.split()
    if len(parts) < 5:
        out.append(line + "\n")
        continue

    cam_id = parts[0]
    model = parts[1]
    width = parts[2]
    height = parts[3]
    params = list(map(float, parts[4:]))

    if len(params) == 3:
        f, cx, cy = params
        fx = fy = f
    elif len(params) >= 4:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        out.append(line + "\n")
        continue

    new_line = f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
    out.append(new_line)

with open(path, "w") as f:
    f.writelines(out)

print("Rewrote camera model to PINHOLE for", path)
