#!/bin/bash
#SBATCH -J chunk_lme
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 6-0
#SBATCH -o /data/delta9043/repos/LightMem/logs/chunk/slurm-%A.out
#SBATCH --exclude=moana-r[1-5],moana-u[1-8]

set -e

echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate lightmem

# ============ 설정 ============
REPO=/data/delta9043/repos/LightMem
DATA=/data/delta9043/datasets/longmemeval/longmemeval_s.json  # TODO: 실제 경로 확인
LLMLINGUA=/data/delta9043/models/llmlingua-2
EMBEDDER=/data/delta9043/models/all-MiniLM-L6-v2
OUTPUT_DIR=${REPO}/results/chunk_longmemeval
# =============================

mkdir -p ${REPO}/logs/chunk ${OUTPUT_DIR}
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

cd ${REPO}/experiments/longmemeval
PYTHONUNBUFFERED=1 python chunk_longmemeval.py \
    --data "$DATA" \
    --llmlingua "$LLMLINGUA" \
    --embedder "$EMBEDDER" \
    --output-dir "$OUTPUT_DIR"

echo "All done: $(date)"
