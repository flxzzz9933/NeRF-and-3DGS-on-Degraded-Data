import sys, os

if len(sys.argv) != 3:
    print("Usage: python filter_sparse_images_to_existing.py <SPARSE_DIR> <IMAGES_DIR>")
    sys.exit(1)

sparse_dir = sys.argv[1]
images_dir = sys.argv[2]

images_txt = os.path.join(sparse_dir, "images.txt")
if not os.path.isfile(images_txt):
    raise FileNotFoundError(images_txt)

existing = set(os.listdir(images_dir))
print(f"[INFO] Found {len(existing)} image files in {images_dir}")

with open(images_txt, "r") as f:
    lines = f.readlines()

out_lines = []
i = 0
kept = 0
dropped = 0

while i < len(lines):
    line = lines[i]
    # 注释或空行，原样保留
    if line.startswith("#") or line.strip() == "":
        out_lines.append(line)
        i += 1
        continue

    header = line.strip().split()
    # COLMAP images.txt 正常 header 至少 10 个字段
    if len(header) < 10:
        out_lines.append(line)
        i += 1
        continue

    image_name = header[-1]
    # 下一行是 2D 特征
    if i + 1 < len(lines):
        pts_line = lines[i + 1]
    else:
        pts_line = ""

    if image_name in existing:
        out_lines.append(line)
        out_lines.append(pts_line)
        kept += 1
    else:
        dropped += 1

    i += 2

print(f"[INFO] Kept {kept} images, dropped {dropped} images without files")

backup = images_txt + ".bak"
os.rename(images_txt, backup)
with open(images_txt, "w") as f:
    f.writelines(out_lines)

print(f"[DONE] Wrote filtered images.txt, original backed up as {backup}")
