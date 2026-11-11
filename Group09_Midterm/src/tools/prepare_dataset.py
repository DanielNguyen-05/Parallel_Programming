"""
prepare_dataset.py - Build the dataset/ folder from the team's raw photos.
CSC14120 Parallel Programming - Final Project - Group 09
Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
         Nguyen Hai Dang (23127165)
Written by the team (with the help of an AI assistant). Uses Pillow and NumPy.

Input layout (one folder per pair, exactly 2 photos, left photo first in
alphabetical order, or named left.* / right.*):
    raw_photos/<pair_name>/<left>.jpg, <right>.jpg      (HEIC works if pillow-heif is installed)
Output layout (same scene, three sizes, cropped to 16:9):
    dataset/<pair_name>/{small,medium,large}/A.jpg, B.jpg
    small = 1280x720, medium = 1920x1080, large = 3840x2160

Usage: python3 prepare_dataset.py raw_photos dataset
"""
import os
import sys
from PIL import Image, ImageOps

try:  # optional HEIC support (iPhone photos)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

SIZES = {"small": (1280, 720), "medium": (1920, 1080), "large": (3840, 2160)}
EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def crop_16_9(img):
    """Largest centered 16:9 crop."""
    w, h = img.size
    if w * 9 > h * 16:          # too wide -> crop the width
        nw = h * 16 // 9
        left = (w - nw) // 2
        return img.crop((left, 0, left + nw, h))
    nh = w * 9 // 16            # too tall -> crop the height
    top = (h - nh) // 2
    return img.crop((0, top, w, top + nh))


def find_pair(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    named = {os.path.splitext(f)[0].lower(): f for f in files}
    if "left" in named and "right" in named:
        return named["left"], named["right"]
    if len(files) != 2:
        raise ValueError(f"{folder}: expected exactly 2 photos, found {len(files)}")
    return files[0], files[1]


def main(raw_dir, out_dir):
    for pair in sorted(os.listdir(raw_dir)):
        folder = os.path.join(raw_dir, pair)
        if not os.path.isdir(folder):
            continue
        left, right = find_pair(folder)
        for role, name in (("A", left), ("B", right)):
            img = ImageOps.exif_transpose(Image.open(os.path.join(folder, name))).convert("RGB")
            src_size = img.size
            img = crop_16_9(img)
            for size_name, size in SIZES.items():
                dst = os.path.join(out_dir, pair, size_name)
                os.makedirs(dst, exist_ok=True)
                img.resize(size, Image.LANCZOS).save(os.path.join(dst, role + ".jpg"), quality=95)
            print(f"{pair}: {role} = {name} {src_size} -> 16:9 crop {img.size} -> {list(SIZES.values())}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
