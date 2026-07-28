#!/bin/bash
# =============================================================================
# Launcher for the RxBrain UnifiedMoT training pipeline.
# Works for BOTH single-node and multi-node:
#   * Single node:  just run  `bash train/launch.sh`
#   * Multi node:   your job system sets WORLD_SIZE / RANK / GPU_NUM /
#                   MASTER_ADDR / MASTER_PORT; they are forwarded to accelerate
#                   and OVERRIDE whatever is in the accelerate yaml, so the same
#                   script self-adapts to any node count.
# =============================================================================
set -e

# ---- Edit these ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_CONFIG="${REPO_ROOT}/train/configs/example.yaml"
ACCELERATE_CONFIG="${REPO_ROOT}/train/configs/deepspeed_zero2.yaml"
TRAIN_SCRIPT="${REPO_ROOT}/train/finetune.py"

# ---- CPU / DataLoader stability --------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1

# ---- Distributed params (from job env, with single-node defaults) ----------
NUM_NODES=${WORLD_SIZE:-1}
GPUS_PER_NODE=${GPU_NUM_PER_NODE:-8}
NODE_RANK=${RANK:-${INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NUM_PROCESSES=${GPU_NUM:-$((NUM_NODES * GPUS_PER_NODE))}

echo "============================================================"
echo "[HOST=$(hostname)] NODE_RANK=${NODE_RANK}/${NUM_NODES}  procs=${NUM_PROCESSES}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}  CONFIG=${TRAINING_CONFIG}"
echo "============================================================"

cd "${REPO_ROOT}"
OUTPUT_DIR_REL=$(grep -E "^  output_dir:" "${TRAINING_CONFIG}" | head -1 | awk '{print $2}')
STDERR_DIR="${REPO_ROOT}/${OUTPUT_DIR_REL}/stderr"
mkdir -p "${STDERR_DIR}"
STDERR_FILE="${STDERR_DIR}/node_${NODE_RANK}.stderr"
export TORCH_DISABLE_ADDR2LINE=1   # surface real C++/HIP errors instead of a symbolize hang

set +e

# ---- GPU preflight: fail fast + name the node if it has no visible GPU ------
# Some stacks probe the GPU at `import transformers` time, so a GPU-less node
# dies deep inside an import with a misleading error. Check up front.
PREFLIGHT_MSG=$(python3 - <<'PY' 2>&1
import sys
try:
    import torch
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError("torch.cuda.device_count()==0")
    print(f"OK gpus={n} name0={torch.cuda.get_device_name(0)}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(42)
PY
)
PREFLIGHT_CODE=$?
echo "[preflight] HOST=$(hostname) NODE_RANK=${NODE_RANK}: ${PREFLIGHT_MSG}" | tee -a "${STDERR_FILE}"
if [ ${PREFLIGHT_CODE} -ne 0 ]; then
  echo "[FATAL] NODE_RANK=${NODE_RANK} HOST=$(hostname): no usable GPU — aborting before launch." | tee -a "${STDERR_FILE}"
  exit 42
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines "${NUM_NODES}" \
  --num_processes "${NUM_PROCESSES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config "${TRAINING_CONFIG}" 2>&1 | tee -a "${STDERR_FILE}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "[exit] code=${TRAIN_EXIT_CODE} (137 = OOM/SIGKILL)  NODE_RANK=${NODE_RANK}"
exit ${TRAIN_EXIT_CODE}
