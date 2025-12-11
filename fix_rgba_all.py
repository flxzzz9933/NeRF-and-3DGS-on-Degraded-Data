import os
from PIL import Image

ROOT = "nerf_data/nerf_synthetic_degraded"

def convert_rgba(root):
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                full = os.path.join(dirpath, fn)
                try:
                    img = Image.open(full)
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                        img.save(full)
                        count += 1
                        if count % 200 == 0:
                            print(f"[INFO] Converted {count} images so far...")
                except:
                    print(f"[WARN] Failed {full}")
    print(f"[DONE] Total converted: {count}")

if __name__ == "__main__":
    convert_rgba(ROOT)
