# 生成 docs/3D肝脏分割部署流程.md 所用的插图
#
# 输入：data/source/ct_liver/（测试切片）、data/source/result/doc_*（trt-volume-seg / liver3d 产物）
# 输出：docs/assets/liver3d/*.png
#
# 用法（工程根目录）:
#   deps/pyenv/bin/python docs/tools/make_liver3d_figs.py
#
# 先按文档第 4/5 节重跑体数据分割，产出 doc_amos_trt / doc_amos_ov / doc_kaggle_seq128 三个结果目录。

import glob
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "data", "source")
OUT = os.path.join(ROOT, "docs", "assets", "liver3d")

# matplotlib 缓存目录（~/.config 不可写时会在工程根目录留下 matplotlib-XXXX 临时目录）
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "build", "mplconfig"))

AMOS_DIR = os.path.join(SRC, "result", "doc_amos_trt")
KAGGLE_DIR = os.path.join(SRC, "result", "doc_kaggle_seq128")
VISTA_DIR = os.path.join(SRC, "result", "spleen_03.nii_vista3d")
SPLEEN_IMG_DIR = os.path.join(SRC, "result", "spleen03_post3d")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org=(8, 24), scale=0.8):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(img, (0, 0), (int(11 * scale * len(text)) + 16, int(30 * scale) + 12), (0, 0, 0), -1)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def overlay(ct, mask, color=(0, 0, 255), alpha=0.4, contour=True):
    """CT 灰度图 + mask 半透明叠加（mask 区域着色，轮廓描白边）"""
    bgr = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
    m = (mask > 0).astype(np.uint8) * 255
    blend = bgr.copy()
    blend[m > 0] = (blend[m > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return blend


def grid(panels, cols, pad=6):
    """panels: 已带标签的 BGR 图列表 -> 拼接成网格"""
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    rows = (len(panels) + cols - 1) // cols
    canvas = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + p.shape[0], x:x + p.shape[1]] = p
    return canvas


def read_raw_volume(path, w, h, d):
    arr = np.fromfile(path, dtype=np.uint8)
    assert arr.size == w * h * d, f"{path}: {arr.size} != {w * h * d}"
    return arr.reshape(d, h, w)


def slice_pngs(pattern):
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------- 1. LiTS 留出切片（单切面链路）
def fig_lits():
    for z in (280, 282):
        src = cv2.imread(os.path.join(SRC, "ct_liver", f"lits_v{z}.tif_z{z}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(SRC, "ct_liver", f"lits_v{z}.tif_z{z}_gt.png"), cv2.IMREAD_GRAYSCALE)
        res_dir = os.path.join(SRC, "result")
        ov = cv2.imread(os.path.join(res_dir, f"lits_v{z}.tif_z{z}-liver-fp32.png"), cv2.IMREAD_COLOR)
        mk = cv2.imread(os.path.join(res_dir, f"lits_v{z}.tif_z{z}-liver-fp32-mask.png"), cv2.IMREAD_GRAYSCALE)
        d = dice(mk, gt)
        panels = [
            label(src, f"input z={z}"),
            label(gt, "LiTS GT"),
            label(ov, f"framework pred (Dice {d:.3f})"),
        ]
        cv2.imwrite(os.path.join(OUT, f"lits.z{z}.triple.png"), grid(panels, 3))


def dice(a, b):
    A, B = a > 127, b > 127
    s = A.sum() + B.sum()
    return 0.0 if s == 0 else 2.0 * np.logical_and(A, B).sum() / s


# ---------------------------------------------------------------- 2. AMOS 体数据：输入 / 逐层 mask / 后处理
def fig_amos():
    ct_dir = os.path.join(SRC, "ct_liver", "amos_volume")
    W, H, D = 512, 512, 201
    raw = read_raw_volume(os.path.join(AMOS_DIR, "mask_volume.raw"), W, H, D)
    fin = read_raw_volume(os.path.join(AMOS_DIR, "post3d_out", "liver_final.u8"), W, H, D)

    zs = [0, 40, 80, 120, 160, 200]
    cv2.imwrite(os.path.join(OUT, "amos.ct-strip.png"), grid(
        [label(cv2.imread(os.path.join(ct_dir, f"amos_z{z:03d}.png"), cv2.IMREAD_GRAYSCALE),
               f"CT z={z}") for z in zs], 3))

    cv2.imwrite(os.path.join(OUT, "amos.raw-mask-strip.png"), grid(
        [label(raw[z], f"raw mask z={z}  {int((raw[z] > 0).sum())} px") for z in zs], 3))

    zs = [100, 160, 194]
    panels = []
    for z in zs:
        ct = cv2.imread(os.path.join(ct_dir, f"amos_z{z:03d}.png"), cv2.IMREAD_GRAYSCALE)
        panels += [
            label(raw[z], f"z={z} raw mask {int((raw[z] > 0).sum())} px"),
            label(fin[z], f"z={z} post3d {int(fin[z].sum())} px"),
            label(overlay(ct, fin[z]), f"z={z} CT + post3d"),
        ]
    cv2.imwrite(os.path.join(OUT, "amos.post3d-before-after.png"), grid(panels, 3))

    # 逐层面积曲线（summary.txt）
    plot_volume_curve(os.path.join(AMOS_DIR, "summary.txt"),
                      os.path.join(OUT, "amos.slice-curve.png"),
                      "AMOS volume: liver pixels per slice (2D UNet, TRT FP32)")


def plot_volume_curve(summary_path, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, px = [], []
    with open(summary_path) as f:
        next(f)
        for line in f:
            cols = line.split("\t")
            if len(cols) < 2 or not cols[1].lstrip("-").isdigit():
                continue
            if int(cols[1]) < 0:
                continue
            z.append(int(cols[0]))
            px.append(int(cols[1]))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(z, px, lw=1.2)
    ax.set_xlabel("slice index (z)")
    ax.set_ylabel("liver pixels")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 3. liver3d PGM 预览（跨层 mask 快照）
def fig_previews():
    def render(out_png, preview_dir):
        pats = sorted(glob.glob(os.path.join(preview_dir, "liver_preview_*.pgm")))
        if not pats:
            return
        cv2.imwrite(os.path.join(OUT, out_png), grid(
            [label(cv2.imread(p, cv2.IMREAD_GRAYSCALE), os.path.basename(p).replace(".pgm", ""))
             for p in pats], 3))

    render("liver3d.previews-amos.png", os.path.join(AMOS_DIR, "post3d_out"))
    render("liver3d.previews-vista3d.png", os.path.join(VISTA_DIR, "post3d_out"))


# ---------------------------------------------------------------- 4. Kaggle 序列（跨域失败案例）
def fig_kaggle():
    W, H, D = 256, 256, 36
    raw = read_raw_volume(os.path.join(KAGGLE_DIR, "mask_volume.raw"), W, H, D)
    fin = read_raw_volume(os.path.join(KAGGLE_DIR, "post3d_out", "liver_final.u8"), W, H, D)
    panels = []
    for z in (6, 12, 18):
        ct = cv2.imread(os.path.join(SRC, "ct_liver", "kaggle_seq128", f"kaggle_z{z:03d}.png"),
                        cv2.IMREAD_GRAYSCALE)
        ct = cv2.resize(ct, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        panels += [
            label(ct, f"CT z={z}"),
            label(cv2.resize(raw[z], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                  f"raw mask {int((raw[z] > 0).sum())} px"),
            label(cv2.resize(fin[z], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) * 255,
                  f"post3d {int(fin[z].sum())} px"),
        ]
    cv2.imwrite(os.path.join(OUT, "kaggle.before-after.png"), grid(panels, 3))


# ---------------------------------------------------------------- 5. VISTA-3D 路径（3D 基础模型）
def fig_vista():
    final_dir = os.path.join(VISTA_DIR, "post3d_out", "final_slices")
    files = slice_pngs(os.path.join(final_dir, "final_slice_*.png"))
    if not files:
        return
    panels = []
    for z in (10, 20, 30):
        ct = cv2.imread(os.path.join(SPLEEN_IMG_DIR, f"image_{z:04d}.png"), cv2.IMREAD_GRAYSCALE)
        mk = cv2.imread(files[z], cv2.IMREAD_GRAYSCALE)
        panels += [label(ct, f"CT z={z}"), label(mk, f"VISTA-3D mask z={z}"),
                   label(overlay(ct, mk), f"overlay z={z}")]
    cv2.imwrite(os.path.join(OUT, "vista3d.spleen03.png"), grid(panels, 3))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_lits()
    fig_amos()
    fig_previews()
    fig_kaggle()
    fig_vista()
    for f in sorted(glob.glob(os.path.join(OUT, "*.png"))):
        im = cv2.imread(f)
        print(f"{os.path.relpath(f, ROOT)}  {im.shape[1]}x{im.shape[0]}")


if __name__ == "__main__":
    main()
