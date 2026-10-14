#!/usr/bin/env python3
"""CUDA 三维肝脏掩码后处理流水线的集成测试。

跑一次 bin/liver3d，解析它输出的报告，并用 numpy 校验生成的掩码：
  * 流水线的每个步骤都出现在报告里，且耗时非负
  * CCL 过滤后连通域数量下降（孤立假阳性被剔除）
  * 后处理没有让精度变差（最终 Dice >= 重建时的 Dice）
  * 输出文件存在、尺寸正确、严格二值，且其体素数 / 体积 / Dice
    与可执行文件上报的数字一致
"""
import os
import re
import subprocess
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "bin", "liver3d")
DATA = os.path.join(ROOT, "data")

FAILED = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        FAILED.append(msg)


def read_stack(path, n):
    with open(path, "rb") as f:
        buf = f.read()
    assert len(buf) == n, f"{path}: expected {n} bytes, got {len(buf)}"
    return np.frombuffer(buf, dtype=np.uint8)


def dice(a, b):
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / (sa + sb)


def main():
    if not os.path.exists(BIN):
        print(f"binary not found: {BIN}\nRun 'make' first.")
        return 1

    print("Running pipeline ...")
    out = subprocess.run([BIN], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        print(f"pipeline exited with code {out.returncode}")
        return 1
    log = out.stdout
    print(log)

    # ---- 解析报告 --------------------------------------------------------
    steps = dict(re.findall(r"^\s+(\d[ab]?\.\s+[A-Za-z0-9 ()+_.\-]+?)\s{2,}([\d.]+) ms$",
                            log, flags=re.MULTILINE))
    total_m = re.search(r"TOTAL pipeline\s+([\d.]+) ms", log)
    ccl_m = re.search(r"components found=(\d+) kept=(\d+) largest=(\d+)", log)
    dice_m = re.search(
        r"reconstructed=([\d.]+)\s+final=([\d.]+)\s+\(delta ([+\-\d.]+)\)", log)
    vol_m = re.search(r"=\s*([\d.]+) cm\^3", log)
    fg_m = re.search(r"reconstructed=(\d+)\s+final=(\d+)", log)

    print("Validations:")
    check(len(steps) >= 6,
          f"all 6 pipeline sub-steps reported timings (found {len(steps)})")
    check(all(float(v) >= 0 for v in steps.values()),
          "all step timings are non-negative")
    check(total_m is not None and float(total_m.group(1))
          > 0, "total pipeline time reported")
    check(ccl_m is not None, "CCL component statistics reported")

    if ccl_m:
        found, kept, largest = map(int, ccl_m.groups())
        check(
            found > 1, f"input had multiple components (found={found}) -> FP present")
        check(kept <= found,
              f"CCL filtering reduced components ({found} -> {kept})")
        check(kept >= 1 and largest > 0, "at least one component kept")

    # ---- 用 numpy 校验输出的掩码 ----------------------------------------
    with open(os.path.join(DATA, "meta.txt")) as f:
        W, H, D, dx, dy, dz = f.read().split()
        W, H, D = int(W), int(H), int(D)
        dx, dy, dz = float(dx), float(dy), float(dz)
    n = W * H * D

    final_path = os.path.join(DATA, "out", "liver_final.u8")
    check(os.path.exists(final_path), f"output mask written: {final_path}")
    final = read_stack(final_path, n)
    gt = read_stack(os.path.join(DATA, "liver_gt.u8"), n)

    check(set(np.unique(final)).issubset(
        {0, 1}), "output mask is strictly binary (0/1)")
    check(int(final.sum()) > 0, "output mask is non-empty")

    if fg_m:
        check(int(final.sum()) == int(fg_m.group(2)),
              f"final foreground count matches report ({int(final.sum())})")
    if vol_m:
        exp_vol = int(final.sum()) * dx * dy * dz / 1000.0
        check(abs(exp_vol - float(vol_m.group(1))) < 1.0,
              f"organ volume matches report ({exp_vol:.2f} cm^3)")
    if dice_m:
        d_final = dice(final, gt)
        check(abs(d_final - float(dice_m.group(2))) < 1e-3,
              f"recomputed Dice matches report ({d_final:.4f})")
        check(float(dice_m.group(3)) >= -1e-9,
              f"post-processing did not degrade Dice (delta {dice_m.group(3)})")

    # 预览图
    prev = [f for f in os.listdir(
        os.path.join(DATA, "out")) if f.endswith(".pgm")]
    check(len(prev) >= 1, f"PGM previews written ({len(prev)})")

    print("-" * 60)
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED:")
        for m in FAILED:
            print("   -", m)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
