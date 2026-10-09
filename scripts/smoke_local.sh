#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-smoke"

echo "==> Configuring Release build..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release

echo "==> Building MemVanta..."
cmake --build "${BUILD_DIR}" --target memvanta memvanta_gguf_inspect memvanta_tests -j2

echo "==> Running tests..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "==> Generating tiny GGUF fixture..."
python3 "${ROOT_DIR}/scripts/make_tiny_gguf.py" \
  "${BUILD_DIR}/smoke_tiny.gguf"

echo "==> Inspecting generated GGUF..."
"${BUILD_DIR}/memvanta_gguf_inspect" \
  "${BUILD_DIR}/smoke_tiny.gguf"

echo
echo "Smoke test completed successfully."
echo "Next step: try the normal MemVanta build and development workflow."
