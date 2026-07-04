#!/usr/bin/bash
#SBATCH -J longmem-chunker-simplemem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 2-0
#SBATCH -o /data/delta9043/repos/SimpleMem/logs/slurm-longmemeval-chunker-simplemem-%A.out

# ── 실행 시간 측정 시작 ──────────────────────────────────
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SECONDS=0

echo "=========================================="
echo "Job started at: ${START_TIME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Experiment: LightMem chunker_v1 for SimpleMem LongMemEval-S"
echo "Input : /data/delta9043/repos/SimpleMem/chunker_inputs/simplemem_longmemeval_chunk.json"
echo "Output: /data/delta9043/repos/SimpleMem/chunker_outputs/simplemem_longmemeval_chunker_v1"
echo "=========================================="

# ── 환경 초기화 ──────────────────────────────────────────
source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate lightmem

export NLTK_DATA=/data/delta9043/nltk_data

# ── 작업 디렉토리 ─────────────────────────────────────────
cd /data/delta9043/repos/LightMem

# ── 로그/입력/출력 디렉토리 보장 ─────────────────────────
mkdir -p /data/delta9043/repos/SimpleMem/logs
mkdir -p /data/delta9043/repos/SimpleMem/chunker_inputs
mkdir -p /data/delta9043/repos/SimpleMem/chunker_outputs/simplemem_longmemeval_chunker_v1

# ── 입력 파일 존재 확인 ─────────────────────────────────
INPUT_FILE="/data/delta9043/repos/SimpleMem/chunker_inputs/simplemem_longmemeval_chunk.json"
OUTPUT_DIR="/data/delta9043/repos/SimpleMem/chunker_outputs/simplemem_longmemeval_chunker_v1"

echo "=========================================="
echo "[Input Check]"
if [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}"
  exit 1
fi

ls -lh "${INPUT_FILE}"

python - <<'PY'
import json
from pathlib import Path

path = Path("/data/delta9043/repos/SimpleMem/chunker_inputs/simplemem_longmemeval_chunk.json")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"num_items={len(data)}")
print(f"first_question_id={data[0].get('question_id')}")
print(f"first_num_sessions={len(data[0].get('haystack_sessions', []))}")
print(f"first_num_dates={len(data[0].get('haystack_dates', []))}")
PY
echo "=========================================="

# ── chunker 코드 경로 설정 확인 ─────────────────────────
echo "=========================================="
echo "[Chunker Script Check]"
grep -n "DATA_PATH\|RESULTS_DIR" experiments/longmemeval/chunker_simplemem.py
echo "=========================================="

# ── 기존 output 확인. 삭제하지 않음 ─────────────────────
# 주의:
# chunks_*.jsonl은 timestamp로 저장되므로 기존 결과를 삭제하지 않는다.
# 필요하면 수동으로 삭제:
# rm -rf /data/delta9043/repos/SimpleMem/chunker_outputs/simplemem_longmemeval_chunker_v1/*

echo "Existing output files:"
ls -lh "${OUTPUT_DIR}" || true

# ── 실행 ─────────────────────────────────────────────────
python experiments/longmemeval/chunker_simplemem.py

EXIT_CODE=$?

# ── 결과 확인 ────────────────────────────────────────────
echo "=========================================="
echo "[Output Check]"
ls -lh "${OUTPUT_DIR}" || true

python - <<'PY'
import json
import glob
from pathlib import Path

out_dir = Path("/data/delta9043/repos/SimpleMem/chunker_outputs/simplemem_longmemeval_chunker_v1")
paths = sorted(glob.glob(str(out_dir / "chunks_*.jsonl")))

print(f"num_chunk_files={len(paths)}")

if paths:
    latest = Path(paths[-1])
    print(f"latest_chunk_file={latest}")

    n = 0
    total_segments = 0

    with open(latest, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_segments += len(record.get("segments", []))

            if n == 0:
                print(f"first_question_id={record.get('question_id')}")
                print(f"first_num_segments={record.get('num_segments')}")
                if record.get("segments"):
                    print(f"first_segment_num_messages={len(record['segments'][0].get('messages', []))}")
                    print(f"first_message_keys={list(record['segments'][0]['messages'][0].keys())}")

            n += 1

    print(f"num_records={n}")
    print(f"total_segments={total_segments}")
    print(f"avg_segments_per_item={total_segments / n if n else 0:.2f}")

    if n != 500:
        print("WARNING: Expected 500 LongMemEval-S records, but got", n)
else:
    print("ERROR: No chunks_*.jsonl file found.")
PY
echo "=========================================="

# ── 실행 시간 출력 ───────────────────────────────────────
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$SECONDS

echo "=========================================="
echo "Job finished at: ${END_TIME}"
echo "Exit code: ${EXIT_CODE}"
echo "Elapsed time: ${ELAPSED} seconds"
echo "Elapsed time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "=========================================="

exit ${EXIT_CODE}