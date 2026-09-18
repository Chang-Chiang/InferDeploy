#!/usr/bin/env bash
# End-to-end: build -> generate test data -> run CUDA pipeline -> validate.
# Extra arguments are forwarded to the pipeline binary, e.g.:
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
