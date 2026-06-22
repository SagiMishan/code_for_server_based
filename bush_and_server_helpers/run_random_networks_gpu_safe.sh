#!/bin/bash
# Runs random_networks.py across the 4 local GPUs, safely coordinating with
# OTHER NODES and OTHER GPUS running this same script at the same time.
#
# Usage:
#   bash run_random_networks_gpu_safe.sh            # idx 0-9
#   bash run_random_networks_gpu_safe.sh 0 4        # idx 0-4 only
#
# Run this on AS MANY NODES AS YOU WANT simultaneously.

START=${1:-0}
END=${2:-9}
N_GPUS=4

BASE_LOG_DIR=/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_logs
BASE_CLAIM_DIR=/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_claims

# Create per-idx log and claim subdirectories upfront for every idx in range
IDX=$START
while [ "$IDX" -le "$END" ]; do
    mkdir -p "$BASE_LOG_DIR/idx_${IDX}"
    mkdir -p "$BASE_CLAIM_DIR/$IDX"
    IDX=$((IDX + 1))
done

NODE_TAG=$(hostname)

export START END BASE_LOG_DIR BASE_CLAIM_DIR NODE_TAG N_GPUS

run_gpu_slot () {
    local GPU_ID=$1
    local IDX=$((START + GPU_ID))
    while [ "$IDX" -le "$END" ]; do
        echo "[$NODE_TAG GPU $GPU_ID] Working on idx $IDX"
        CUDA_VISIBLE_DEVICES=$GPU_ID SLURM_ARRAY_TASK_ID=$IDX \
            python -u code_compare_results/random_networks.py \
            >> "$BASE_LOG_DIR/idx_${IDX}/${NODE_TAG}_gpu${GPU_ID}.out" \
            2>> "$BASE_LOG_DIR/idx_${IDX}/${NODE_TAG}_gpu${GPU_ID}.err"
        IDX=$((IDX + N_GPUS))
    done
    echo "[$NODE_TAG GPU $GPU_ID] No more idx in range $START-$END. Exiting."
}

for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    nohup bash -c "$(declare -f run_gpu_slot); run_gpu_slot $GPU_ID" \
        > "$BASE_LOG_DIR/${NODE_TAG}_gpu${GPU_ID}_random_slot.out" 2>&1 &
    disown
    echo "Launched $NODE_TAG GPU $GPU_ID random-networks slot manager (PID $!)"
done

echo ""
echo "All random-networks GPU slot managers launched on $NODE_TAG for idx range $START-$END."
echo "Safe to run this same script on other nodes/GPUs simultaneously."
echo ""
echo "Log structure:   $BASE_LOG_DIR/idx_<N>/<node>_gpu<G>.out"
echo "Claim structure: $BASE_CLAIM_DIR/<N>/<hex_name>.claim/"
echo ""
echo "Monitor with:"
echo "  tail -f $BASE_LOG_DIR/idx_*/*.out"
echo "  bash bush_and_server_helpers/manage_random_claims.sh status <idx>"
echo "  nvidia-smi"