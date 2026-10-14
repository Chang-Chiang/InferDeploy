#!/usr/bin/env python3
"""把真实的分割掩码（例如 UNet 的肝脏预测）导入流水线需要的输入格式。

接受一个存放 2D 切面掩码的目录，格式可以是 PNG/JPG/TIFF（需要 Pillow）、
PGM/PBM（PNM，仅用 numpy 解码），或无头的 RAW 字节文件，然后转成：

    <out>/slices/slice_%04d.u8   每个切面一个 W*H 字节的 0/1 文件
    <out>/meta.txt               "W H D dx dy dz"

例子（UNet 预测存成了 0/255 的 PNG，512x512，面内 1mm，层厚 2mm）：
    python3 tools/import_masks.py --indir unet_pred_png --out data \
        --dx 0.7 --dy 0.7 --dz 2.0 --ext '*.png' --threshold 127

可以跑 UNet（或直接复用）的公开肝脏 CT 数据集：
    3D-IRCADb-01、LiTS（Liver Tumor Segmentation Challenge）。
"""
import argparse
import glob
import os
import re
import sys
import numpy as np


def read_pnm(path, thr, invert):
    """仅用 numpy 把 PGM/PBM（P1/P2/P4/P5）解码成 0/1 的 uint8 数组。"""
    with open(path, "rb") as f:
        buf = f.read()
    n = len(buf)
    pos = 0
    ws = (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C)

    def token():
        nonlocal pos
        while pos < n:
            c = buf[pos]
            if c == 0x23:                       # '#' 开头是文件内的注释行
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

    if magic == b"P5":                          # 二进制灰度
        pos += 1
        data = np.frombuffer(buf[pos:pos + w * h],
                             dtype=np.uint8).reshape(h, w)
        arr = ((data.astype(np.uint16) * 255 // max(maxv, 1))
               > thr).astype(np.uint8)
    elif magic == b"P4":                        # 二进制位图（1 表示黑色）
        pos += 1
        rowbytes = (w + 7) // 8
        bits = np.unpackbits(
            np.frombuffer(buf[pos:pos + rowbytes * h],
                          dtype=np.uint8).reshape(h, rowbytes),
            axis=1)
        arr = bits[:, :w].astype(np.uint8)
    elif magic in (b"P1", b"P2"):               # ASCII 位图 / 灰度图
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
    # 光栅格式交给 Pillow 处理（可选依赖）
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
