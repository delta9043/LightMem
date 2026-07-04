#!/usr/bin/bash
#SBATCH -J lightmem-chunk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/delta9043/repos/LightMem/logs/chunk-%A.out
#SBATCH -e /data/delta9043/repos/LightMem/logs/chunk-%A.err

set -euo pipefail

echo "=== Job Info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv

# conda 활성화
source /data/delta9043/anaconda3/etc/profile.d/conda.sh
conda activate lightmem

cd /data/delta9043/repos/LightMem
which python
python --version

# 실행
python experiments/longmemeval/see_chunks.py

echo "=== Done ==="
date
exit 0