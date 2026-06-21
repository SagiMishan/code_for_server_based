#!/bin/bash
# Runs all 100 realisations (or whatever range you specify) across 4 GPUs,
# automatically starting the next task as soon as a GPU frees up.
#
# Usage:
#   ./run_all_gpu.sh            # runs tasks 0-99
#   ./run_all_gpu.sh 0 19       # runs tasks 0-19 only
#
# Logs go to simulation/log_files/run_<task_id>.out / .err
# Survives SSH disconnect (uses nohup under the hood via disown).

START=${1:-0}
END=${2:-99}
N_GPUS=4

mkdir -p simulation/log_files

# One "slot" per GPU. We launch a background loop per GPU that pulls
# the next task ID from a shared counter file.
echo $START > simulation/log_files/.next_task_id

run_gpu_slot () {
    local GPU_ID=$1
    while true; do
        # atomically grab next task id
        TASK_ID=$(flock simulation/log_files/.next_task_id.lock -c "
            CUR=\$(cat simulation/log_files/.next_task_id)
            if [ \$CUR -gt $END ]; then
                echo -1
            else
                echo \$((CUR + 1)) > simulation/log_files/.next_task_id
                echo \$CUR
            fi
        ")
        if [ "$TASK_ID" -lt 0 ]; then
            echo "[GPU $GPU_ID] No more tasks. Exiting."
            break
        fi
        echo "[GPU $GPU_ID] Starting realisation $TASK_ID"
        CUDA_VISIBLE_DEVICES=$GPU_ID SLURM_ARRAY_TASK_ID=$TASK_ID \
            python main.py > simulation/log_files/run_${TASK_ID}.out 2> simulation/log_files/run_${TASK_ID}.err
        echo "[GPU $GPU_ID] Finished realisation $TASK_ID"
    done
}

for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    nohup bash -c "$(declare -f run_gpu_slot); run_gpu_slot $GPU_ID" \
        > simulation/log_files/gpu${GPU_ID}_slot.out 2>&1 &
    disown
    echo "Launched GPU $GPU_ID slot manager (PID $!)"
done

echo ""
echo "All GPU slot managers launched. They will keep training realisations"
echo "$START..$END until all are done. You can safely close this terminal."
echo ""
echo "Monitor with:"
echo "  tail -f simulation/log_files/run_*.out"
echo "  nvidia-smi"
echo "  ps aux | grep main.py"