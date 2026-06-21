#!/bin/bash
# Checks GPU specs + current load across all candidate GPU nodes.
# Run this from dsihead (or wherever you have SSH access to all nodes).
#
# Usage: bash check_all_gpus.sh

NODES=(
    dsigpu01
    dsigpu02
    dsigpu03
    dsigpu04
    dsigpu05
    dsicsgpu02
    dsicsgpu03
    dsicsgpu04
    dsicsgpu07
    dsicsgpu08
    dsicsgpu09
    dsicsgpu10
)

echo "================================================================"
echo " GPU NODE COMPARISON"
echo "================================================================"

for NODE in "${NODES[@]}"; do
    echo ""
    echo "── $NODE ──────────────────────────────────────────────"
    timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" '
        nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu,temperature.gpu \
            --format=csv,noheader 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "  [unreachable or no GPU]"
        fi
    ' 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  [SSH timeout / unreachable]"
    fi
done

echo ""
echo "================================================================"
echo " Columns: GPU_idx, Name, Compute_Cap, Total_Mem, Used_Mem, Util%, Temp"
echo " Lower 'Used_Mem' and 'Util%' = more free capacity right now"
echo " Higher 'Compute_Cap' = newer architecture (more PyTorch features)"
echo "================================================================"
