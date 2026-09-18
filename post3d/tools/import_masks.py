#!/usr/bin/env python3
"""Import real segmentation masks (e.g. UNet liver predictions) into the
pipeline's input format.

Accepts a directory of 2D slice masks as PNG/JPG/TIFF (needs Pillow), PGM/PBM
(PNM, decoded with numpy only), or headerless RAW bytes, and converts them into:

    <out>/slices/slice_%04d.u8   one W*H byte file per slice (0/1)
    <out>/meta.txt               "W H D dx dy dz"

Example (UNet predictions saved as 0/255 PNGs, 512x512, 1mm in-plane, 2mm slices):
    python3 tools/import_masks.py --indir unet_pred_png --out data \
        --dx 0.7 --dy 0.7 --dz 2.0 --ext '*.png' --threshold 127

Public liver CT datasets whose masks you can run a UNet on (or reuse):
    3D-IRCADb-01, LiTS (Liver Tumor Segmentation Challenge).
"""
import argparse
import glob
import os
import re
import sys
import numpy as np


def read_pnm(path, thr, invert):
    """Decode PGM/PBM (P1/P2/P4/P5) to a 0/1 uint8 array using numpy only."""
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf)
    pos = 0
    ws = (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C)

    def token():
        nonlocal pos
        while pos < n:
            c = buf[pos]
            if c == 0x23:                       # '#' comment
                while pos < n and buf[pos] != 0x0A:
                    pos += 1
                continue
            if c in ws:
                pos += 1
                continue
            break
        s = pos
        while pos < n and buf[pos] not in ws:
            pos += 1
        return buf[s:pos]

    magic = token()
    w = int(token())
    h = int(token())
    maxv = 1
    if magic in (b"P2", b"P5"):
        maxv = int(token())

    if magic == b"P5":                          # binary grayscale
        pos += 1
        data = np.frombuffer(buf[pos:pos + w * h],
                             dtype=np.uint8).reshape(h, w)
        arr = ((data.astype(np.uint16) * 255 // max(maxv, 1))
               > thr).astype(np.uint8)
    elif magic == b"P4":                        # binary bitmap (1 = black)
        pos += 1
        rowbytes = (w + 7) // 8
        bits = np.unpackbits(
            np.frombuffer(buf[pos:pos + rowbytes * h],
                          dtype=np.uint8).reshape(h, rowbytes),
            axis=1)
        arr = bits[:, :w].astype(np.uint8)
    elif magic in (b"P1", b"P2"):               # ascii bitmap / grayscale
        vals = []
        while len(vals) < w * h:
            vals.append(int(token()))
        a = np.array(vals, dtype=np.uint16).reshape(h, w)
        arr = ((a * 255 // max(maxv, 1)) >
               thr).astype(np.uint8) if magic == b"P2" else a.astype(np.uint8)
    else:
        raise ValueError(f"unsupported PNM magic {magic!r} in {path}")

    return (1 - arr) if invert else arr


def read_image(path, thr, invert, rawW, rawH):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pgm", ".pbm", ".pnm", ".ppm"):
        return read_pnm(path, thr, invert)
    if ext in (".raw", ".u8", ".bin"):
        if rawW <= 0 or rawH <= 0:
            raise ValueError("RAW slices require --W and --H")
        with open(path, "rb") as f:
            buf = f.read(rawW * rawH)
        a = (np.frombuffer(buf, dtype=np.uint8).reshape(
            rawH, rawW) > thr).astype(np.uint8)
        return (1 - a) if invert else a
    # raster formats via Pillow (optional)
    from PIL import Image  # noqa
    im = Image.open(path).convert("L")
    a = (np.array(im) > thr).astype(np.uint8)
    return (1 - a) if invert else a


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(s))]


def main():
    ap = argparse.ArgumentParser(
        description="Import 2D mask slices into pipeline input format")
    ap.add_argument("--indir", required=True,
                    help="directory of 2D slice masks")
    ap.add_argument("--out", default="data",
                    help="output data dir (default ./data)")
    ap.add_argument("--ext", default="*",
                    help="glob for slice files (e.g. '*.png')")
    ap.add_argument("--W", type=int, default=0,
                    help="slice width (required for RAW)")
    ap.add_argument("--H", type=int, default=0,
                    help="slice height (required for RAW)")
    ap.add_argument("--D", type=int, default=0,
                    help="number of slices (default: all found)")
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--dz", type=float, default=1.0)
    ap.add_argument("--threshold", type=int, default=127,
                    help="foreground where value > threshold")
    ap.add_argument("--invert", action="store_true",
                    help="invert mask polarity")
    args = ap.parse_args()

    pat = args.ext if "*" in args.ext else "*" + args.ext
    files = sorted(glob.glob(os.path.join(args.indir, pat)), key=natural_key)
    if args.D > 0:
        files = files[:args.D]
    if not files:
        print(f"no slice files matched in {args.indir} (pattern '{pat}')")
        return 1

    slices_dir = os.path.join(args.out, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out, "out"), exist_ok=True)

    W = H = None
    for z, fp in enumerate(files):
        a = read_image(fp, args.threshold, args.invert, args.W, args.H)
        if W is None:
            H, W = a.shape
        elif a.shape != (H, W):
            print(f"[import] size mismatch at {fp}: {a.shape} vs {(H, W)}")
            return 1
        with open(os.path.join(slices_dir, f"slice_{z:04d}.u8"), "wb") as f:
            f.write(a.astype(np.uint8).tobytes())

    D = len(files)
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"{W} {H} {D} {args.dx} {args.dy} {args.dz}\n")

    print(
        f"[import] {D} slices  {W}x{H}  spacing {args.dx}x{args.dy}x{args.dz} mm")
    print(
        f"[import] wrote -> {slices_dir}/slice_*.u8 and {os.path.join(args.out, 'meta.txt')}")
    print("[import] run:  ./bin/liver3d --meta " + os.path.join(args.out, "meta.txt") +
          " --slices " + slices_dir + " --no-gt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
