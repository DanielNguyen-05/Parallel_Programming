"""
make_synthetic.py - Build a synthetic image pair with a known (ground-truth) homography.
CSC14120 Parallel Programming - Final Project - Group 09
Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
         Nguyen Hai Dang (23127165)
Written by the team (with the help of an AI assistant). Uses Pillow and NumPy.

Idea: a camera that only ROTATES produces two images related exactly by the
homography H = K R K^-1 (K = intrinsics, R = rotation). We treat a photo as the
scene seen by camera A, render image A as a window of that photo, and render
image B as the view of the same camera turned by yaw (plus a small pitch/roll).
Every pixel of B is mapped into A's frame with H (B -> A, the convention of the
stitching program) and sampled from the photo with bilinear interpolation.

Output: <out>/<size>/A.jpg, B.jpg, H_gt.txt for small / medium / large
Usage:  python3 make_synthetic.py <source_photo> <out_dir> [yaw_deg]
"""
import os
import sys
import numpy as np
from PIL import Image, ImageOps

SIZES = {"small": (1280, 720), "medium": (1920, 1080), "large": (3840, 2160)}
FOCAL = 0.9          # focal length in units of the image width (~58 degree horizontal FOV)
PITCH, ROLL = 1.5, 2.0  # degrees


def rotation(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def bilinear(img, xs, ys):
    """Sample img (H x W x 3 float) at float coordinates; returns values and validity."""
    h, w = img.shape[:2]
    valid = (xs >= 0) & (ys >= 0) & (xs <= w - 1) & (ys <= h - 1)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[..., None]; fy = (ys - y0)[..., None]
    v = (1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1]) + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1])
    return v, valid


def main(src, out_dir, yaw):
    photo = np.asarray(ImageOps.exif_transpose(Image.open(src)).convert("RGB"), dtype=np.float32)
    ph, pw = photo.shape[:2]
    # Window of image A inside the photo: 16:9, 55% of the photo width, near the left side
    a_w = 0.55 * pw
    a_h = a_w * 9 / 16
    x0 = 0.03 * pw
    y0 = (ph - a_h) / 2
    for name, (W, Hh) in SIZES.items():
        k = a_w / W                                    # photo pixels per output pixel
        K = np.array([[FOCAL * W, 0, W / 2], [0, FOCAL * W, Hh / 2], [0, 0, 1]])
        H = K @ rotation(yaw, PITCH, ROLL) @ np.linalg.inv(K)   # B -> A
        H /= H[2, 2]
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(Hh, dtype=np.float64))
        # Image A: window of the photo
        imgA, okA = bilinear(photo, x0 + k * u, y0 + k * v)
        # Image B: map each pixel of B into A's frame with H, then into the photo
        den = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        xa = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / den
        ya = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / den
        imgB, okB = bilinear(photo, x0 + k * xa, y0 + k * ya)
        coverage = okB.mean()
        if coverage < 0.999:
            raise SystemExit(f"Image B is not fully inside the photo ({coverage:.1%}); use a smaller yaw or a wider photo")
        # Overlap: fraction of B's pixels that fall inside A
        overlap = ((xa >= 0) & (xa <= W - 1) & (ya >= 0) & (ya <= Hh - 1)).mean()
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(np.clip(imgA + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(d, "A.jpg"), quality=95)
        Image.fromarray(np.clip(imgB + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(d, "B.jpg"), quality=95)
        np.savetxt(os.path.join(d, "H_gt.txt"), H, fmt="%.12g")
        print(f"{name}: {W}x{Hh}, yaw {yaw} deg, pitch {PITCH}, roll {ROLL}, overlap of B with A = {overlap:.1%}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) == 4 else 20.0)
