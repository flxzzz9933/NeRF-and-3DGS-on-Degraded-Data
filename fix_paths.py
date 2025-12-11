import json
import os
from glob import glob

ROOT = "nerf_data/nerf_synthetic_degraded"

def fix_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    fixed = False
    for frame in data.get("frames", []):
        fp = frame.get("file_path", "")
        if not fp.endswith(".png"):
            frame["file_path"] = fp + ".png"
            fixed = True

    if fixed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[FIXED] {path}")
    else:
        print(f"[OK] No change: {path}")


def main():
    jsons = glob(f"{ROOT}/**/transforms.json", recursive=True)
    print(f"Found {len(jsons)} transforms.json files")
    for js in jsons:
        fix_json(js)

if __name__ == "__main__":
    main()
