#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-or-large-file> [output-dir]" >&2
  exit 2
fi

INPUT="$1"
OUT="${2:-benchmark-evidence/issue12}"
BIN="${MEMVANTA_BIN:-./build/memvanta}"
CHUNK="${CHUNK:-64M}"
CACHE="${CACHE:-512M}"
PASSES="${PASSES:-3}"
REPEATS="${REPEATS:-5}"

mkdir -p "$OUT"
{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "input=$INPUT"
  echo "chunk=$CHUNK"
  echo "cache=$CACHE"
  echo "passes=$PASSES"
  echo "repeats=$REPEATS"
  uname -a
  lscpu || true
  free -h || true
} > "$OUT/environment.txt"

run_case() {
  local name="$1"; shift
  : > "$OUT/${name}.txt"
  for i in $(seq 1 "$REPEATS"); do
    echo "=== run=$i case=$name ===" | tee -a "$OUT/${name}.txt"
    /usr/bin/time -v "$BIN" run "$INPUT" \
      --chunk "$CHUNK" --cache "$CACHE" --passes "$PASSES" "$@" \
      >> "$OUT/${name}.txt" 2>> "$OUT/${name}.txt"
  done
}

# Same input, chunk/cache ceiling, passes, binary, and host for all cases.
run_case baseline --prefetch 0
run_case fixed_prefetch --prefetch 2
run_case bounded_prefetch --prefetch 2
run_case adaptive --prefetch 2 --adaptive-prefetch --prefetch-min 1 --prefetch-max 4 --prefetch-window 8

python3 - "$OUT" <<'PY'
import pathlib, re, statistics, sys
out = pathlib.Path(sys.argv[1])
cases = ["baseline", "fixed_prefetch", "bounded_prefetch", "adaptive"]
rows = []
for case in cases:
    text = (out / f"{case}.txt").read_text()
    throughput = [float(x) for x in re.findall(r"Effective stream: ([0-9.eE+-]+) GiB/s", text)]
    rss = [float(x) for x in re.findall(r"Peak RSS: ([0-9.eE+-]+) MiB", text)]
    checksums = re.findall(r"Checksum: ([0-9]+)", text)
    rows.append((case, statistics.median(throughput), max(rss), len(set(checksums)) == 1, checksums[0] if checksums else ""))
base = rows[0][1]
with (out / "summary.csv").open("w") as f:
    f.write("case,median_gib_s,speedup_vs_baseline,peak_rss_mib,checksum_stable,checksum\n")
    for case, tp, rss, stable, checksum in rows:
        f.write(f"{case},{tp:.6f},{tp/base:.4f},{rss:.3f},{str(stable).lower()},{checksum}\n")
print((out / "summary.csv").read_text())
PY
