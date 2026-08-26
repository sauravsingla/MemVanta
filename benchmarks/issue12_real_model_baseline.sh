#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [output-dir]" >&2
  exit 2
fi

MODEL="$1"
OUT="${2:-benchmark-evidence/issue12-real-model}"
BIN="${MEMVANTA_REAL_BENCH:-./build/memvanta_real_bench}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-1}"
CTX="${CTX:-256}"
PROMPT="${PROMPT:-64}"
GEN="${GEN:-16}"
CLIENT_PROMPT="${CLIENT_PROMPT:-32}"
CLIENT_OUT="${CLIENT_OUT:-8}"
BATCH="${BATCH:-16}"
KV="${KV:-f16}"

mkdir -p "$OUT"
{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "model=$MODEL"
  echo "reps=$REPS warmup=$WARMUP ctx=$CTX prompt=$PROMPT gen=$GEN client_prompt=$CLIENT_PROMPT client_out=$CLIENT_OUT batch=$BATCH kv=$KV"
  uname -a
  lscpu || true
  free -h || true
} > "$OUT/environment.txt"

run_case() {
  local threads="$1"
  local csv="$OUT/t${threads}.csv"
  local log="$OUT/t${threads}.txt"
  /usr/bin/time -v "$BIN" \
    --model "$MODEL" --threads "$threads" --reps "$REPS" --warmup "$WARMUP" \
    --ctx "$CTX" --prompt "$PROMPT" --gen "$GEN" \
    --client-prompt "$CLIENT_PROMPT" --client-out "$CLIENT_OUT" \
    --batch "$BATCH" --kv "$KV" --csv "$csv" \
    > "$log" 2> "$OUT/t${threads}.time.txt"
}

run_case 1
run_case 4

python3 - "$OUT" <<'PY'
import csv, pathlib, re, statistics, sys
out = pathlib.Path(sys.argv[1])
rows=[]
for threads in (1,4):
    samples=list(csv.DictReader((out/f"t{threads}.csv").open()))
    def vals(k): return [float(r[k]) for r in samples if r.get(k)]
    log=(out/f"t{threads}.txt").read_text()
    rss_match=re.search(r"peak_rss=([0-9.eE+-]+) MiB",log)
    if not rss_match: raise SystemExit(f"missing peak_rss for t{threads}")
    rows.append({
        "threads":threads,
        "pp_tps_median":statistics.median(vals("pp_tps")),
        "tg_tps_median":statistics.median(vals("tg_tps")),
        "ttft_ms_median":statistics.median(vals("ttft_ms")),
        "output_tps_median":statistics.median(vals("output_tps")),
        "peak_rss_mib":float(rss_match.group(1)),
    })
with (out/"summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print((out/"summary.csv").read_text())
PY
