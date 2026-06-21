#!/bin/bash
# Runs realisations [START..END] across the 4 local GPUs, safely coordinating
# with OTHER NODES running this same script at the same time.
#
# Safety mechanism: before training realisation N, this script atomically
# creates a claim file at simulation/claims/task_N.claim using `mkdir`
# (mkdir is atomic on POSIX/NFS - only one process can succeed even across
# different machines, as long as they share the same NFS-mounted folder).
# If the claim already exists, this GPU skips that realisation and moves on.
#
# Usage:
#   bash run_all_gpu.sh            # runs tasks 0-99
#   bash run_all_gpu.sh 0 19       # runs tasks 0-19 only
#
# Run this on AS MANY NODES AS YOU WANT simultaneously - they will not
# duplicate work as long as they share the same project folder over NFS.

START=${1:-0}
END=${2:-99}
N_GPUS=4

LOG_DIR=simulation/log_files/compare_networks/logs
CLAIM_DIR=simulation/log_files/compare_networks/claims

mkdir -p "$LOG_DIR"
mkdir -p "$CLAIM_DIR"

NODE_TAG=$(hostname)

# Export everything run_gpu_slot() needs, since it runs inside a separate
# `bash -c` subshell (via nohup) that does NOT inherit unexported variables.
export START END LOG_DIR CLAIM_DIR NODE_TAG

run_gpu_slot () {
    local GPU_ID=$1
    local TASK_ID=$START
    while [ "$TASK_ID" -le "$END" ]; do
        CLAIM_PATH="$CLAIM_DIR/task_${TASK_ID}.claim"

        # Atomic claim: mkdir fails if the directory already exists,
        # even across different machines on the same NFS mount.
        if mkdir "$CLAIM_PATH" 2>/dev/null; then
            # We won the claim - record who/where for debugging
            echo "$NODE_TAG GPU$GPU_ID $(date)" > "$CLAIM_PATH/owner.txt"

            echo "[$NODE_TAG GPU $GPU_ID] Starting realisation $TASK_ID"
            CUDA_VISIBLE_DEVICES=$GPU_ID SLURM_ARRAY_TASK_ID=$TASK_ID \
                python -u main.py \
                > "$LOG_DIR/run_${TASK_ID}.out" \
                2> "$LOG_DIR/run_${TASK_ID}.err"
            echo "[$NODE_TAG GPU $GPU_ID] Finished realisation $TASK_ID"

            # mark as done so it's clear at a glance (optional, claim dir staying = done)
            touch "$CLAIM_PATH/done.txt"
        else
            # Someone else (this node or another node) already claimed it - skip
            :
        fi
        TASK_ID=$((TASK_ID + 1))
    done
    echo "[$NODE_TAG GPU $GPU_ID] No more tasks in range $START-$END. Exiting."
}

for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    nohup bash -c "$(declare -f run_gpu_slot); run_gpu_slot $GPU_ID" \
        > "$LOG_DIR/${NODE_TAG}_gpu${GPU_ID}_slot.out" 2>&1 &
    disown
    echo "Launched $NODE_TAG GPU $GPU_ID slot manager (PID $!)"
done

echo ""
echo "All GPU slot managers launched on $NODE_TAG for range $START-$END."
echo "Safe to run this same script on other nodes simultaneously -"
echo "claims in $CLAIM_DIR/ prevent any realisation from running twice."
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/run_*.out"
echo "  ls $CLAIM_DIR/                 # see which realisations are claimed/done"
echo "  cat $CLAIM_DIR/task_5/owner.txt   # see who claimed task 5"
echo "  nvidia-smi"