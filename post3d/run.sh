#!/usr/bin/env bash
# 端到端流程：构建 -> 生成测试数据 -> 跑 CUDA 流水线 -> 校验结果。
# 额外的参数会透传给流水线可执行文件，例如：
#   ./run.sh --minsize 128 --mrz 2 --sigma 1.5 --repeat 10
set -euo pipefail
cd "$(dirname "$0")"

echo "==================== build ===================="
make -s

echo "================ kernel self-test ================"
./bin/liver3d --selftest

if [[ ! -f data/meta.txt || "${REGEN:-0}" == "1" ]]; then
  echo "================ generate test data ==============="
  python3 tools/gen_test_data.py
else
  echo "(test data present; set REGEN=1 to regenerate)"
fi

echo "================= run pipeline =================="
./bin/liver3d "$@"

echo "=================== validate ===================="
python3 tests/test_pipeline.py
