#!/bin/bash
# Stops ALL running jobs (training, evaluation, random-network search) across
# every GPU node you've used. Run this from dsihead or any node that can SSH
# to the others.
#
# Usage: bash stop_all.sh

NODES=(
    dsicsgpu02
    dsicsgpu03
    dsicsgpu04
    dsicsgpu07
    dsicsgpu08
    dsicsgpu09
    dsicsgpu10
)

for NODE in "${NODES[@]}"; do
    echo "── Stopping jobs on $NODE ──────────────────────────────"
    timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" '
        pkill -9 -f run_gpu_slot 2>/dev/null
        pkill -9 -f main.py 2>/dev/null
        pkill -9 -f /home/dsi/mishans1/projects/code_for_server_based/code_compare_results/evaluate_all_networks.py 2>/dev/null
        pkill -9 -f /home/dsi/mishans1/projects/code_for_server_based/code_compare_results/random_networks.py 2>/dev/null
        echo "  done."
    ' 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  [SSH timeout / unreachable — skipped]"
    fi
done

echo ""
echo "All nodes processed. Verify with:"
echo "  for n in ${NODES[*]}; do echo \"-- \$n --\"; ssh \$n 'ps aux | grep -E \"main.py|evaluate_all_networks.py|random_networks.py\"'; done"
