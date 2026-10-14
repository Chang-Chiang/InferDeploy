#!/usr/bin/env python3
"""生成合成的 UNet 风格肝脏分割测试数据（仅依赖 numpy）。

产生一份干净的肝脏金标准掩码，以及一份*预测*掩码，后者刻意模拟逐切面
2D UNet 推理的典型缺陷：

  * 抖动 / 不稳定的边界（逐切面推理噪声）
  * 层间不连续（若干切面被平移或丢弃 -> 形成薄缝隙）
  * 内部孔洞（血管被误判为背景）
  * 边缘毛刺（表面上细弱的伪突出）
  * 孤立假阳性（只出现在 1-2 个切面的小连通域）

输出（位于 --outdir，默认 ./data）：
  slices/slice_%04d.u8   逐切面预测掩码（流水线输入），每个 W*H 字节
  meta.txt               "W H D dx dy dz"
  liver_pred_stack.u8    合并后的预测栈（W*H*D 字节）
  liver_gt.u8            干净的金标准栈（用于 Dice 上报）
  preview/*.pgm          若干轴位 PGM 预览图（pred 与 gt）
"""
import argparse
import os
import numpy as np


# ----------------------------- 小工具函数 ----------------------------------
def gaussian_blur3(vol, sigma):
    """可分离的三维高斯模糊，边界用复制填充（不依赖 scipy）。"""
    r = max(1, int(np.ceil(3 * sigma)))
    ax = np.arange(-r, r + 1)
    k = np.exp(-(ax ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    out = vol.astype(np.float32)
    for axis in range(3):
        out = np.moveaxis(out, axis, 0)
        pad = np.pad(out, ((r, r), (0, 0), (0, 0)), mode="edge")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * pad[i:i + out.shape[0]]
        out = np.moveaxis(acc, 0, axis)
    return out


def erode6(m):
    e = m.copy()
    for ax in range(3):
        e &= np.roll(m, 1, ax) & np.roll(m, -1, ax)
    return e


def dilate6(m):
    d = m.copy()
    for ax in range(3):
        d |= np.roll(m, 1, ax) | np.roll(m, -1, ax)
    return d


def set_ball(vol, cz, cy, cx, r, val):
    D, H, W = vol.shape
    z0, z1 = max(0, cz - r), min(D, cz + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    win = vol[z0:z1, y0:y1, x0:x1]
    win[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val


def set_disc(vol, z, cy, cx, r, val):
    D, H, W = vol.shape
    if not (0 <= z < D):
        return
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    sl = vol[z, y0:y1, x0:x1]
    sl[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val


def save_pgm(path, arr2d):
    h, w = arr2d.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write((arr2d.astype(np.uint8) * 255).tobytes())


# ------------------------------- 肝脏模型 -----------------------------------
def build_liver_gt(W, H, D, rng):
    """由两个椭球拼出的肝脏状团块（右叶 + 左叶），并带一个心切迹。"""
    z, y, x = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

    # 右叶（较大）、左叶（较小，且有偏移）
    crx, cry, crz = 0.42 * W, 0.56 * H, 0.50 * D
    rr = ((x - crx) / (0.25 * W)) ** 2 + ((y - cry) /
                                          (0.22 * H)) ** 2 + ((z - crz) / (0.33 * D)) ** 2

    clx, cly, clz = 0.63 * W, 0.40 * H, 0.47 * D
    rl = ((x - clx) / (0.125 * W)) ** 2 + ((y - cly) /
                                           (0.11 * H)) ** 2 + ((z - clz) / (0.25 * D)) ** 2

    v = np.minimum(rr, rl)

    # 心切迹：挖出一个凹区，使形状不只是一个完美团块
    nx, ny, nz = 0.55 * W, 0.30 * H, 0.30 * D
    notch = ((x - nx) / (0.14 * W)) ** 2 + ((y - ny) /
                                            (0.12 * H)) ** 2 + ((z - nz) / (0.20 * D)) ** 2
    v = np.where(notch < 1.0, np.maximum(v, 1.0 + 0.35 * (1.0 - notch)), v)

    gt = (v <= 1.0).astype(np.uint8)
    return gt, v


def make_prediction(gt, v, W, H, D, rng):
    pred = gt.copy()

    # 1) 边界抖动：对隐式曲面的阈值做局部扰动
    noise = gaussian_blur3(rng.standard_normal(
        (D, H, W)).astype(np.float32), sigma=2.5)
    noise /= (np.abs(noise).max() + 1e-6)
    pred = (v <= 1.0 + 0.22 * noise).astype(np.uint8)

    # 2) 层间不连续：平移若干切面，并抹掉一小块区域
    shift_slices = rng.choice(D, size=max(3, D // 16), replace=False)
    for zs in shift_slices:
        dy = int(rng.integers(-2, 3))
        dx = int(rng.integers(-2, 3))
        pred[zs] = np.roll(pred[zs], (dy, dx), axis=(0, 1))
    # 再丢掉几个切面 -> 形成必须由形态学闭运算弥合的缝隙
    for zs in rng.choice(D, size=3, replace=False):
        cy, cx = int(0.5 * H), int(0.45 * W)
        pred[zs, cy - 10:cy + 10, cx - 14:cx + 14] = 0

    # 3) 内部孔洞（血管）
    interior = erode6(erode6(gt.astype(bool)))
    iz, iy, ix = np.nonzero(interior)
    if len(iz):
        for idx in rng.choice(len(iz), size=14, replace=False):
            set_ball(pred, int(iz[idx]), int(iy[idx]), int(
                ix[idx]), int(rng.integers(1, 5)), 0)

    # 4) 边缘毛刺：表面上的细弱突出
    surface = gt.astype(bool) & ~erode6(gt.astype(bool))
    sz, sy, sx = np.nonzero(surface)
    if len(sz):
        for idx in rng.choice(len(sz), size=600, replace=False):
            cz, cy, cx = int(sz[idx]), int(sy[idx]), int(sx[idx])
            dz, dy, dx = rng.integers(-1, 2, size=3)
            for t in range(1, int(rng.integers(2, 6))):
                nz_, ny_, nx_ = cz + dz * t, cy + dy * t, cx + dx * t
                if 0 <= nz_ < D and 0 <= ny_ < H and 0 <= nx_ < W:
                    pred[nz_, ny_, nx_] = 1

    # 5) 孤立假阳性（小块，有些只出现在单个切面）
    placed = 0
    tries = 0
    while placed < 90 and tries < 8000:
        tries += 1
        cz = int(rng.integers(4, D - 4))
        cy = int(rng.integers(6, H - 6))
        cx = int(rng.integers(6, W - 6))
        # 要求邻域干净，这样这个假阳性才是真正孤立的
        if pred[max(0, cz - 5):cz + 6, max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6].any():
            continue
        if rng.random() < 0.5:
            set_disc(pred, cz, cy, cx, int(
                rng.integers(1, 3)), 1)  # 单切面假阳性
        else:
            set_ball(pred, cz, cy, cx, int(
                rng.integers(1, 3)), 1)  # 跨几个切面的假阳性
        placed += 1

    return pred


# ---------------------------------- 主流程 ----------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Synthetic UNet liver-mask test data generator")
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dy", type=float, default=1.0)
    ap.add_argument("--dz", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    W, H, D = args.W, args.H, args.D
    rng = np.random.default_rng(args.seed)

    print(f"[gen] building liver GT {W}x{H}x{D} ...")
    gt, v = build_liver_gt(W, H, D, rng)
    print("[gen] synthesizing UNet-like prediction artifacts ...")
    pred = make_prediction(gt, v, W, H, D, rng)

    out = args.outdir
    slices_dir = os.path.join(out, "slices")
    prev_dir = os.path.join(out, "preview")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "out"), exist_ok=True)

    # 逐切面掩码（逐层推理流水线的输入）
    for z in range(D):
        with open(os.path.join(slices_dir, f"slice_{z:04d}.u8"), "wb") as f:
            f.write(pred[z].astype(np.uint8).tobytes())

    # 合并后的数据栈 + 元数据
    with open(os.path.join(out, "liver_pred_stack.u8"), "wb") as f:
        f.write(pred.astype(np.uint8).tobytes())
    with open(os.path.join(out, "liver_gt.u8"), "wb") as f:
        f.write(gt.astype(np.uint8).tobytes())
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"{W} {H} {D} {args.dx} {args.dy} {args.dz}\n")

    # 预览图（中间切面 + 两个偏心切面），pred 与 gt 各一份
    for z in (D // 4, D // 2, (3 * D) // 4):
        save_pgm(os.path.join(prev_dir, f"pred_z{z:03d}.pgm"), pred[z])
        save_pgm(os.path.join(prev_dir, f"gt_z{z:03d}.pgm"), gt[z])

    voxel_mm3 = args.dx * args.dy * args.dz
    gt_vol_cm3 = gt.sum() * voxel_mm3 / 1000.0
    pred_vol_cm3 = pred.sum() * voxel_mm3 / 1000.0
    inter = np.logical_and(gt, pred).sum()
    dice = 2 * inter / (gt.sum() + pred.sum())
    print("[gen] --------------------------------------------------")
    print(
        f"[gen] GT   foreground voxels = {gt.sum():>9d}  volume = {gt_vol_cm3:8.2f} cm^3")
    print(
        f"[gen] PRED foreground voxels = {pred.sum():>9d}  volume = {pred_vol_cm3:8.2f} cm^3")
    print(f"[gen] PRED vs GT Dice (raw)  = {dice:.4f}")
    print(f"[gen] wrote slices -> {slices_dir}/slice_*.u8  ({D} files)")
    print(
        f"[gen] wrote stacks -> {out}/liver_pred_stack.u8, {out}/liver_gt.u8, {out}/meta.txt")
    print(f"[gen] wrote previews -> {prev_dir}/*.pgm")


if __name__ == "__main__":
    main()
