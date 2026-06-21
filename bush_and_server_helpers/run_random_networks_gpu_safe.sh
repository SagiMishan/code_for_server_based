#!/bin/bash
# Runs random_networks.py across the 4 local GPUs, safely coordinating with
# OTHER NODES and OTHER GPUS running this same script at the same time.
#
# Two layers of safety:
#   1. GPUs are assigned non-overlapping idx via round-robin offset (no two
#      GPUs attempt the same main-model idx redundantly).
#   2. Within an idx, random_networks.py itself claims each individual random
#      selection atomically (simulation/random_claims/<idx>/<name>.claim),
#      so multiple GPUs/nodes CAN work on the same idx concurrently without
#      ever training the same random selection twice.
#
# Usage:
#   bash run_random_networks_gpu_safe.sh            # idx 0-9
#   bash run_random_networks_gpu_safe.sh 0 4         # idx 0-4 only
#
# Run this on AS MANY NODES AS YOU WANT simultaneously.

START=${1:-0}
END=${2:-9}
N_GPUS=4

LOG_DIR=/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_log
CLAIM_DIR=/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_claims

mkdir -p "$LOG_DIR"
mkdir -p "$CLAIM_DIR"

NODE_TAG=$(hostname)
 
export START END LOG_DIR CLAIM_DIR NODE_TAG N_GPUS
 
run_gpu_slot () {
    local GPU_ID=$1
    local IDX=$((START + GPU_ID))
    while [ "$IDX" -le "$END" ]; do
        echo "[$NODE_TAG GPU $GPU_ID] Working on idx $IDX"
        CUDA_VISIBLE_DEVICES=$GPU_ID SLURM_ARRAY_TASK_ID=$IDX \
            python -u code_compare_results/random_networks.py \
            >> "$LOG_DIR/idx_${IDX}.out" \
            2>> "$LOG_DIR/idx_${IDX}.err"
        IDX=$((IDX + N_GPUS))
    done
    echo "[$NODE_TAG GPU $GPU_ID] No more idx in range $START-$END. Exiting."
}
 
for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    nohup bash -c "$(declare -f run_gpu_slot); run_gpu_slot $GPU_ID" \
        > "$LOG_DIR/${NODE_TAG}_gpu${GPU_ID}_random_slot.out" 2>&1 &
    disown
    echo "Launched $NODE_TAG GPU $GPU_ID random-networks slot manager (PID $!)"
done
 
echo ""
echo "All random-networks GPU slot managers launched on $NODE_TAG for idx range $START-$END."
echo "Safe to run this same script on other nodes/GPUs simultaneously - claims in"
echo "simulation/random_claims/<idx>/ prevent any random selection from being trained twice."
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/idx_*.out"
echo "  bash manage_random_claims.sh status <idx>"
echo "  nvidia-smi"
 