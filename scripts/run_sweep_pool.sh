#!/bin/bash
# =============================================================================
# Continuous Slot-Filling Hyperparameter Sweep (IBM Power / High-CPU Server)
# =============================================================================
# Replaces run_cpu_sweep.sh's batch-wait with continuous slot-filling:
# two independent pools (GPU + CPU) tracked by PID. When ANY single job
# finishes, the next one starts immediately. No idle slots.
#
# Designed for the IBM Power server where CPU overflow jobs take ~30min
# while GPU jobs take ~5min — batch-wait wastes 59 slots waiting for 1 slow job.
# For the GPU-only server, use run_cube_sweep.sh (batch-wait is fine there).
#
# Requires bash 4.3+ (for wait -n). Phase 2 only: cached data required.
#
# Usage:
#   # Full sweep with all participants:
#   PARTICIPANTS="010012 010016 010017 010019 010020 010021 010022 010023 010024 010026" \
#     MAX_CPU_WORKERS=80 nohup ./run_sweep_pool.sh > sweep_pool.log 2>&1 &
#
#   # Quick test (small grid, dry run):
#   ./run_sweep_pool.sh --dry-run --participants "010012 010016" \
#     --k-range "3 4 5" --ld-range "32" --depth-range "3" --ndf-range "64"
#
#   # GPU selection:
#   GPU_IDS="1 3" ./run_sweep_pool.sh                    # Only GPUs 1 and 3
#   GPU_EXCLUDE="0" ./run_sweep_pool.sh                  # Skip GPU 0
#
#   # Override sweep grid:
#   K_RANGE="3 4 5" LD_RANGE="32" ./run_sweep_pool.sh
# =============================================================================

set -euo pipefail

# Prevent BLAS/OpenMP thread conflicts across concurrent processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Configuration ---
PARTICIPANTS="${PARTICIPANTS:-010012 010016 010017 010019 010020 010021 010022 010023 010024 010026}"
MAX_CPU_WORKERS="${MAX_CPU_WORKERS:-20}"
STAGGER_SECS="${STAGGER_SECS:-1}"
RESERVE_MB="${RESERVE_MB:-4000}"
PER_JOB_MB="${PER_JOB_MB:-1500}"
DRY_RUN=0
PREPROCESS=0

# Sweep grid (overridable via env vars)
K_RANGE="${K_RANGE:-$(seq 3 20)}"
LD_RANGE="${LD_RANGE:-16 32 64}"
DEPTH_RANGE="${DEPTH_RANGE:-2 3 4}"
NDF_RANGE="${NDF_RANGE:-32 64 128}"

# --- CLI Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=1; shift ;;
        --preprocess)     PREPROCESS=1; shift ;;
        --participants)   PARTICIPANTS="$2"; shift 2 ;;
        --max-cpu)        MAX_CPU_WORKERS="$2"; shift 2 ;;
        --k-range)        K_RANGE="$2"; shift 2 ;;
        --ld-range)       LD_RANGE="$2"; shift 2 ;;
        --depth-range)    DEPTH_RANGE="$2"; shift 2 ;;
        --ndf-range)      NDF_RANGE="$2"; shift 2 ;;
        --stagger)        STAGGER_SECS="$2"; shift 2 ;;
        --reserve-mb)     RESERVE_MB="$2"; shift 2 ;;
        --per-job-mb)     PER_JOB_MB="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Bash Version Check ---
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3))); then
    echo "ERROR: Requires bash 4.3+ for 'wait -n'. Found: $BASH_VERSION"
    echo "  Both Linux servers have this. macOS ships bash 3.2 (sweeps don't run on Mac)."
    exit 1
fi

# --- Environment Detection ---
PYTHON_CMD=()

detect_environment() {
    local conda_dir="$HOME/miniconda3"
    if [ -f "$conda_dir/bin/conda" ]; then
        # IBM Power / conda environment
        set +u
        export PATH="$conda_dir/bin:$PATH"
        eval "$($conda_dir/bin/conda shell.bash hook)"
        conda activate sweep
        set -u
        PYTHON_CMD=(python3)
        echo "Environment: conda (IBM Power)"
    elif [ -f ".venv-py36/bin/activate" ]; then
        source ".venv-py36/bin/activate"
        PYTHON_CMD=(python3)
        echo "Environment: venv-py36"
    elif command -v poetry &>/dev/null; then
        PYTHON_CMD=(poetry run python)
        echo "Environment: Poetry"
    else
        PYTHON_CMD=(python3)
        echo "Environment: system python3"
    fi
}

# --- GPU Capacity (per-GPU slot tracking) ---
declare -A GPU_MAX_SLOTS    # gpu_id → max concurrent jobs
declare -A GPU_RUNNING      # gpu_id → currently running count
GPU_IDS_LIST=()             # ordered list of usable GPU IDs
GPU_TOTAL_SLOTS=0

compute_gpu_capacity() {
    GPU_IDS_LIST=()
    GPU_MAX_SLOTS=()
    GPU_RUNNING=()
    GPU_TOTAL_SLOTS=0

    if ! command -v nvidia-smi &>/dev/null; then
        echo "  No nvidia-smi found — GPU pool disabled"
        return
    fi

    while IFS=',' read -r idx free; do
        idx="${idx// /}"; free="${free// /}"
        # Include filter
        if [ -n "${GPU_IDS:-}" ]; then
            local match=0
            for g in $GPU_IDS; do [ "$idx" = "$g" ] && match=1; done
            [ "$match" -eq 0 ] && continue
        fi
        # Exclude filter
        local skip=0
        for ex in ${GPU_EXCLUDE:-}; do [ "$idx" = "$ex" ] && skip=1; done
        [ "$skip" -eq 1 ] && continue
        # Compute slots from available memory
        local available=$((free - RESERVE_MB))
        if [ "$available" -le "$PER_JOB_MB" ]; then
            echo "  GPU $idx: 0 slots (${free}MB free, need ${RESERVE_MB}MB reserve)"
            continue
        fi
        local slots=$((available / PER_JOB_MB))
        GPU_IDS_LIST+=("$idx")
        GPU_MAX_SLOTS[$idx]=$slots
        GPU_RUNNING[$idx]=0
        GPU_TOTAL_SLOTS=$((GPU_TOTAL_SLOTS + slots))
        echo "  GPU $idx: $slots slots (${free}MB free - ${RESERVE_MB}MB reserve = ${available}MB usable)"
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)

    echo "  Total GPU slots: $GPU_TOTAL_SLOTS | CPU slots: $MAX_CPU_WORKERS"
}

# --- Pool Tracking ---
# Note: bash 4.3 + set -u treats empty associative arrays as unbound.
# Use integer counters (gpu_pool_active, cpu_pool_active) for size checks
# instead of ${#gpu_pool[@]} / ${#cpu_pool[@]}.
declare -A gpu_pool       # pid → gpu_id
declare -A cpu_pool       # pid → 1
declare -A job_labels     # pid → label string
gpu_pool_active=0
cpu_pool_active=0
gpu_launched=0
cpu_launched=0
gpu_completed=0
cpu_completed=0
total_queued=0

# --- Pool Functions ---

pool_status() {
    local gpu_active=0
    if [ ${#GPU_IDS_LIST[@]} -gt 0 ]; then
        for gid in "${GPU_IDS_LIST[@]}"; do
            gpu_active=$((gpu_active + ${GPU_RUNNING[$gid]}))
        done
    fi
    echo "GPU:${gpu_active}/${GPU_TOTAL_SLOTS} CPU:${cpu_pool_active}/${MAX_CPU_WORKERS}"
}

launch_job() {
    local participant="$1" k="$2" ld="$3" depth="$4" ndf="$5" device="$6" gpu_id="${7:-}"
    local label="${k}_128_${ld}_${depth}_${ndf}"
    local run_id="sweep_4d_${participant}"
    local log_file="logs/${participant}_k${k}_ld${ld}_d${depth}_w${ndf}.log"

    local cuda_env=""
    [ "$device" = "gpu" ] && cuda_env="$gpu_id"

    CUDA_VISIBLE_DEVICES="$cuda_env" "${PYTHON_CMD[@]}" train.py \
        --participant "$participant" \
        --n_clusters "$k" \
        --latent_dim "$ld" \
        --n_conv_layers "$depth" \
        --ndf "$ndf" \
        --run_id "$run_id" \
        --use-cached \
        --fast-sweep \
        >"$log_file" 2>&1 &

    local pid=$!
    job_labels[$pid]="${participant}/${label}"

    if [ "$device" = "gpu" ]; then
        gpu_pool[$pid]="$gpu_id"
        gpu_pool_active=$((gpu_pool_active + 1))
        GPU_RUNNING[$gpu_id]=$(( ${GPU_RUNNING[$gpu_id]} + 1 ))
        gpu_launched=$((gpu_launched + 1))
        echo "[$(date '+%H:%M:%S')] GPU $gpu_id: $label [$(pool_status)]"
    else
        cpu_pool[$pid]=1
        cpu_pool_active=$((cpu_pool_active + 1))
        cpu_launched=$((cpu_launched + 1))
        echo "[$(date '+%H:%M:%S')] CPU:    $label [$(pool_status)]"
    fi
    total_queued=$((total_queued + 1))
}

try_launch_gpu() {
    if [ ${#GPU_IDS_LIST[@]} -eq 0 ]; then return 1; fi
    for gpu_id in "${GPU_IDS_LIST[@]}"; do
        if [ "${GPU_RUNNING[$gpu_id]}" -lt "${GPU_MAX_SLOTS[$gpu_id]}" ]; then
            launch_job "$1" "$2" "$3" "$4" "$5" "gpu" "$gpu_id"
            return 0
        fi
    done
    return 1
}

try_launch_cpu() {
    if [ "$cpu_pool_active" -lt "$MAX_CPU_WORKERS" ]; then
        launch_job "$1" "$2" "$3" "$4" "$5" "cpu"
        return 0
    fi
    return 1
}

reap_one() {
    # Block until any child exits
    wait -n 2>/dev/null || true

    # Scan GPU pool for the finished PID
    local pid
    if [ "$gpu_pool_active" -gt 0 ]; then
        for pid in "${!gpu_pool[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local gpu_id="${gpu_pool[$pid]}"
                local label="${job_labels[$pid]:-?}"
                GPU_RUNNING[$gpu_id]=$(( ${GPU_RUNNING[$gpu_id]} - 1 ))
                unset "gpu_pool[$pid]"
                unset "job_labels[$pid]"
                gpu_pool_active=$((gpu_pool_active - 1))
                gpu_completed=$((gpu_completed + 1))
                echo "[$(date '+%H:%M:%S')] DONE GPU $gpu_id: $label [$(pool_status)]"
                return 0
            fi
        done
    fi

    # Scan CPU pool for the finished PID
    if [ "$cpu_pool_active" -gt 0 ]; then
        for pid in "${!cpu_pool[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local label="${job_labels[$pid]:-?}"
                unset "cpu_pool[$pid]"
                unset "job_labels[$pid]"
                cpu_pool_active=$((cpu_pool_active - 1))
                cpu_completed=$((cpu_completed + 1))
                echo "[$(date '+%H:%M:%S')] DONE CPU:    $label [$(pool_status)]"
                return 0
            fi
        done
    fi

    # wait -n returned but no tracked PID was found (shouldn't happen)
    return 1
}

drain_pools() {
    local remaining=$(( gpu_pool_active + cpu_pool_active ))
    if [ "$remaining" -eq 0 ]; then return; fi
    echo "[$(date '+%H:%M:%S')] Draining $remaining remaining jobs..."
    while [ "$gpu_pool_active" -gt 0 ] || [ "$cpu_pool_active" -gt 0 ]; do
        reap_one || sleep 1  # safety: if reap finds nothing, avoid tight loop
    done
}

# --- Cleanup on Interrupt ---
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Interrupted! Sending SIGTERM to background jobs..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Killed. GPU completed: $gpu_completed | CPU completed: $cpu_completed"
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Main
# =============================================================================

detect_environment
mkdir -p logs

echo ""
echo "============================================================"
echo "CONTINUOUS POOL SWEEP (Phase 2: Cached Data)"
echo "============================================================"
echo "Participants: $PARTICIPANTS"
echo "K: $K_RANGE"
echo "Latent dims: $LD_RANGE | Depths: $DEPTH_RANGE | NDF widths: $NDF_RANGE"
echo "Max CPU workers: $MAX_CPU_WORKERS | Stagger: ${STAGGER_SECS}s"
echo "Reserve: ${RESERVE_MB}MB/GPU | Per-job estimate: ${PER_JOB_MB}MB"
[ "$DRY_RUN" -eq 1 ] && echo "MODE: DRY RUN (no jobs will be launched)"
compute_gpu_capacity

# Sanity check: at least one slot available
if [ "$GPU_TOTAL_SLOTS" -eq 0 ] && [ "$MAX_CPU_WORKERS" -eq 0 ]; then
    echo "ERROR: No GPU or CPU slots available. Check GPU_IDS/GPU_EXCLUDE/MAX_CPU_WORKERS."
    exit 1
fi

echo "Python: $("${PYTHON_CMD[@]}" --version 2>&1)"
echo "============================================================"

SWEEP_START=$(date +%s)

for PARTICIPANT in $PARTICIPANTS; do
    RUN_ID="sweep_4d_${PARTICIPANT}"
    CACHE_DIR="data/cache/$PARTICIPANT"

    # Optional preprocessing
    if [ "$PREPROCESS" -eq 1 ]; then
        echo ""
        echo "[$(date '+%H:%M:%S')] Preprocessing: $PARTICIPANT"
        "${PYTHON_CMD[@]}" train.py \
            --participant "$PARTICIPANT" \
            --n_clusters 3 \
            --run_id "$RUN_ID" \
            --preprocess-only
    fi

    # Verify cache exists
    if [ ! -f "$CACHE_DIR/all_data.npy" ] || \
       [ ! -f "$CACHE_DIR/norm_params.npz" ] || \
       [ ! -f "$CACHE_DIR/split_indices.npz" ]; then
        echo "ERROR: Cache missing for $PARTICIPANT in $CACHE_DIR/"
        echo "  Run with --preprocess or transfer cache first."
        continue
    fi

    echo ""
    echo "============================================================"
    echo "SWEEP: Participant $PARTICIPANT"
    echo "============================================================"

    local_queued=0
    local_skipped=0
    for k in $K_RANGE; do
        for ld in $LD_RANGE; do
            for depth in $DEPTH_RANGE; do
                for ndf in $NDF_RANGE; do
                    LABEL="${k}_128_${ld}_${depth}_${ndf}"
                    OUTPUT_DIR="outputs/${PARTICIPANT}/${RUN_ID}/cluster_${LABEL}"
                    MARKER="$OUTPUT_DIR/summary_metrics.yaml"

                    # Skip completed jobs
                    if [ -f "$MARKER" ]; then
                        local_skipped=$((local_skipped + 1))
                        continue
                    fi

                    # Dry run: just print
                    if [ "$DRY_RUN" -eq 1 ]; then
                        echo "  [DRY-RUN] K=$k ld=$ld depth=$depth ndf=$ndf"
                        local_queued=$((local_queued + 1))
                        continue
                    fi

                    # Launch: try GPU first, then CPU, else wait for a slot
                    while true; do
                        if try_launch_gpu "$PARTICIPANT" "$k" "$ld" "$depth" "$ndf"; then break; fi
                        if try_launch_cpu "$PARTICIPANT" "$k" "$ld" "$depth" "$ndf"; then break; fi
                        reap_one || true
                    done

                    local_queued=$((local_queued + 1))
                    sleep "$STAGGER_SECS"
                done
            done
        done
    done

    echo "[$(date '+%H:%M:%S')] $PARTICIPANT: $local_queued queued, $local_skipped skipped (already complete)"
done

# Drain all remaining jobs
if [ "$DRY_RUN" -eq 0 ]; then
    drain_pools
fi

# =============================================================================
# Summary
# =============================================================================
SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
COMPLETED=$(find outputs/ -name "summary_metrics.yaml" 2>/dev/null | wc -l | tr -d ' ')
echo "Total completed configs: $COMPLETED"
echo "This session — GPU: $gpu_completed | CPU: $cpu_completed | Queued: $total_queued"
echo "Wall time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "============================================================"
