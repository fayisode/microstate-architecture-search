#!/bin/bash
#SBATCH --partition=datamite
#SBATCH --gres=gpu:2
#SBATCH --job-name=eeg_4d_sweep
#SBATCH --time=336:00:00
#SBATCH --mem=160G

# =============================================================================
# 4D Hyperparameter Sweep — Tier 1: Fast Screening
# =============================================================================
# Sweeps over K × latent_dim × depth (n_conv_layers) × width (ndf)
# with --fast-sweep mode: 20 epochs, no viz, metrics only.
#
# Usage:
#   ./run_cube_sweep.sh                           # Default participant
#   ./run_cube_sweep.sh --participants "010004 010005"
#   sbatch run_cube_sweep.sh
# =============================================================================

set -euo pipefail

# Prevent BLAS/OpenMP thread conflicts across concurrent processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Configuration ---
PARTICIPANTS="${PARTICIPANTS:-010004}"
MAX_CONCURRENT="${MAX_CONCURRENT:-80}"
STAGGER_SECS=3
RESERVE_MB="${RESERVE_MB:-4000}"     # VRAM headroom per GPU for other processes (xai, etc.)
PER_JOB_MB="${PER_JOB_MB:-1500}"     # Estimated VRAM per fast-sweep job

# Capacity-proportional GPU allocation:
#   - Queries each GPU's free memory, subtracts RESERVE_MB safety buffer
#   - Allocates floor((free - reserve) / per_job) slots per GPU
#   - Jobs beyond GPU_CAPACITY fall back to CPU
#   - Re-queries after each batch wait (adapts as jobs complete / xai changes)
#
# Override:  GPU_IDS="1 3"       — only consider these GPUs
# Exclude:   GPU_EXCLUDE="0 2"   — skip GPUs used by others
compute_gpu_slots() {
    GPU_ASSIGNMENTS=()
    if ! command -v nvidia-smi &>/dev/null; then
        GPU_CAPACITY=0
        return
    fi
    while IFS=',' read -r idx free; do
        idx="${idx// /}"; free="${free// /}"
        # Include filter
        if [ -n "${GPU_IDS:-}" ]; then
            match=0
            for g in $GPU_IDS; do [ "$idx" = "$g" ] && match=1; done
            [ "$match" -eq 0 ] && continue
        fi
        # Exclude filter
        skip=0
        for ex in ${GPU_EXCLUDE:-}; do [ "$idx" = "$ex" ] && skip=1; done
        [ "$skip" -eq 1 ] && continue
        # Compute slots from available memory
        available=$((free - RESERVE_MB))
        if [ "$available" -le "$PER_JOB_MB" ]; then
            echo "  GPU $idx: 0 slots (${free}MB free, need ${RESERVE_MB}MB reserve)"
            continue
        fi
        slots=$((available / PER_JOB_MB))
        for ((s=0; s<slots; s++)); do GPU_ASSIGNMENTS+=("$idx"); done
        echo "  GPU $idx: $slots slots (${free}MB free - ${RESERVE_MB}MB reserve = ${available}MB usable)"
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    GPU_CAPACITY=${#GPU_ASSIGNMENTS[@]}
    echo "  Total GPU slots: $GPU_CAPACITY | Overflow → CPU"
}

# Sweep grid (narrowed for tier 1: drop ld=8 (too small) and ld=128 (OOM-prone))
K_RANGE=$(seq 3 20)
LD_RANGE="16 32 64"
DEPTH_RANGE="2 3 4"
NDF_RANGE="32 64 128"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --participants) PARTICIPANTS="$2"; shift 2 ;;
        --k-range) K_RANGE="$2"; shift 2 ;;
        --ld-range) LD_RANGE="$2"; shift 2 ;;
        --depth-range) DEPTH_RANGE="$2"; shift 2 ;;
        --ndf-range) NDF_RANGE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs

echo "============================================================"
echo "4D HYPERPARAMETER SWEEP — TIER 1 (FAST SCREENING)"
echo "============================================================"
echo "Participants: $PARTICIPANTS"
echo "K: $K_RANGE"
echo "Latent dims: $LD_RANGE"
echo "Depths: $DEPTH_RANGE"
echo "NDF widths: $NDF_RANGE"
echo "Reserve: ${RESERVE_MB}MB/GPU | Per-job estimate: ${PER_JOB_MB}MB"
echo "Max concurrent: $MAX_CONCURRENT"
compute_gpu_slots
echo "============================================================"

# Ensure Poetry is available
if ! command -v poetry &>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v poetry &>/dev/null; then
    echo "ERROR: Poetry command not found. Please install Poetry first."
    exit 1
fi

# Remove stale lock and install dependencies
if [ -f "poetry.lock" ]; then
    echo "Removing existing poetry.lock..."
    rm poetry.lock
fi

echo "Installing dependencies..."
poetry install

for PARTICIPANT in $PARTICIPANTS; do
    RUN_ID="sweep_4d_${PARTICIPANT}"

    echo ""
    echo "============================================================"
    echo "PHASE 1: Preprocessing — Participant $PARTICIPANT"
    echo "============================================================"

    poetry run python train.py \
        --participant "$PARTICIPANT" \
        --n_clusters 3 \
        --run_id "$RUN_ID" \
        --preprocess-only

    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed for $PARTICIPANT!"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "PHASE 2: Sweep — Participant $PARTICIPANT"
    echo "============================================================"

    job_count=0
    gpu_launched=0
    for k in $K_RANGE; do
        for ld in $LD_RANGE; do
            for depth in $DEPTH_RANGE; do
                for ndf in $NDF_RANGE; do

                    # OOM guard: skip configs that are too large for concurrent GPU runs
                    # With 4 processes per GPU, each model+optimizer+batch must fit in ~25% of VRAM
                    if [ "$ndf" -ge 128 ] && [ "$depth" -ge 5 ]; then
                        echo "SKIP: ndf=$ndf depth=$depth (OOM risk — large model)"
                        continue
                    fi
                    if [ "$ndf" -ge 128 ] && [ "$ld" -ge 128 ]; then
                        echo "SKIP: ndf=$ndf ld=$ld (OOM risk — wide latent + wide conv)"
                        continue
                    fi
                    if [ "$depth" -ge 5 ] && [ "$ld" -ge 128 ] && [ "$ndf" -ge 64 ]; then
                        echo "SKIP: depth=$depth ld=$ld ndf=$ndf (OOM risk — deep + wide)"
                        continue
                    fi

                    # Capacity-proportional GPU/CPU assignment
                    if [ "$gpu_launched" -lt "$GPU_CAPACITY" ]; then
                        GPU_ID=${GPU_ASSIGNMENTS[$gpu_launched]}
                        LOG_FILE="logs/sweep_${PARTICIPANT}_k${k}_ld${ld}_d${depth}_w${ndf}_gpu${GPU_ID}.log"
                        echo "[$(date '+%H:%M:%S')] K=$k ld=$ld depth=$depth ndf=$ndf → GPU $GPU_ID"

                        CUDA_VISIBLE_DEVICES=$GPU_ID poetry run python train.py \
                            --participant "$PARTICIPANT" \
                            --n_clusters "$k" \
                            --latent_dim "$ld" \
                            --n_conv_layers "$depth" \
                            --ndf "$ndf" \
                            --run_id "$RUN_ID" \
                            --use-cached \
                            --fast-sweep \
                            >"$LOG_FILE" 2>&1 &
                        gpu_launched=$((gpu_launched + 1))
                    else
                        LOG_FILE="logs/sweep_${PARTICIPANT}_k${k}_ld${ld}_d${depth}_w${ndf}_cpu.log"
                        echo "[$(date '+%H:%M:%S')] K=$k ld=$ld depth=$depth ndf=$ndf → CPU"

                        CUDA_VISIBLE_DEVICES="" poetry run python train.py \
                            --participant "$PARTICIPANT" \
                            --n_clusters "$k" \
                            --latent_dim "$ld" \
                            --n_conv_layers "$depth" \
                            --ndf "$ndf" \
                            --run_id "$RUN_ID" \
                            --use-cached \
                            --fast-sweep \
                            >"$LOG_FILE" 2>&1 &
                    fi

                    job_count=$((job_count + 1))
                    sleep $STAGGER_SECS

                    # Throttle: wait when batch limit is reached
                    if [ $((job_count % MAX_CONCURRENT)) -eq 0 ]; then
                        echo "[$(date '+%H:%M:%S')] Batch of $MAX_CONCURRENT jobs done. Re-querying GPU memory..."
                        wait
                        # Re-query: completed jobs freed VRAM, xai may have changed
                        compute_gpu_slots
                        gpu_launched=0
                    fi

                done
            done
        done
    done

    echo "[$(date '+%H:%M:%S')] All sweep jobs launched for $PARTICIPANT. Waiting..."
    wait
    echo "[$(date '+%H:%M:%S')] Sweep complete for $PARTICIPANT."

    # Aggregation
    echo ""
    echo "Running aggregation..."
    poetry run python aggregate_results.py \
        --run_id "$RUN_ID" \
        --participant "$PARTICIPANT"

    # Sweep analysis
    echo "Running sweep analysis..."
    poetry run python sweep_analysis.py \
        --run_id "$RUN_ID" \
        --participant "$PARTICIPANT"

done

echo ""
echo "============================================================"
echo "4D SWEEP COMPLETE — ALL PARTICIPANTS"
echo "============================================================"
