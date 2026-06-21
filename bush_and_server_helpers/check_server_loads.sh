#!/bin/bash
# Checks load across all candidate GPU nodes and prints them ranked from
# best (most free) to worst. Nodes where YOU already have a job running
# (main.py / evaluate_all_networks.py / random_networks.py / run_gpu_slot)
# are skipped from the ranking, since you shouldn't pile more work onto a
# node you're already using.
#
# Usage: bash check_server_loads.sh

NODES=(
    dsicsgpu02
    dsicsgpu03
    dsicsgpu04
    dsicsgpu07
    dsicsgpu08
    dsicsgpu09
    dsicsgpu10
)

MY_USER=$(whoami)

echo "================================================================"
echo " SERVER LOAD CHECK — as $MY_USER"
echo "================================================================"

declare -A SCORE
declare -A SUMMARY
declare -a SKIPPED=()
declare -a USABLE=()

for NODE in "${NODES[@]}"; do
    RESULT=$(timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$MY_USER@$NODE" '
        # Check if I already have a job running on this node
        MY_JOBS=$(ps -u '"$MY_USER"' -o pid=,pcpu=,pmem=,etime=,cmd= 2>/dev/null | grep -E "main\.py|evaluate_all_networks\.py|random_networks\.py|run_gpu_slot" | grep -v grep)
        MY_JOBS_COUNT=$(echo "$MY_JOBS" | grep -c .)

        if [ "$MY_JOBS_COUNT" -gt 0 ]; then
            echo "SKIP|$MY_JOBS_COUNT job(s) running"
            echo "$MY_JOBS" | while read -r line; do
                echo "JOBLINE|$line"
            done
            exit 0
        fi

        # GPU stats: avg utilization % and avg used memory MiB across all GPUs
        GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ -z "$GPU_STATS" ]; then
            echo "ERROR|nvidia-smi failed or no GPU"
            exit 0
        fi

        N_GPU=$(echo "$GPU_STATS" | wc -l)
        AVG_UTIL=$(echo "$GPU_STATS" | awk -F, "{sum+=\$1} END {printf \"%.1f\", sum/NR}")
        TOTAL_USED=$(echo "$GPU_STATS" | awk -F, "{sum+=\$2} END {print sum}")
        TOTAL_MEM=$(echo "$GPU_STATS" | awk -F, "{sum+=\$3} END {print sum}")

        # CPU load average (1 min)
        LOAD1=$(cut -d" " -f1 /proc/loadavg)
        N_CPU=$(nproc)

        # Other users GPU memory usage (everyone, including yourself, in MiB)
        OTHER_MEM=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk "{sum+=\$1} END {print sum+0}")

        echo "OK|$AVG_UTIL|$TOTAL_USED|$TOTAL_MEM|$N_GPU|$LOAD1|$N_CPU|$OTHER_MEM"
    ' 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$RESULT" ]; then
        echo ""
        echo "── $NODE: [UNREACHABLE] ──────────────────────────────"
        continue
    fi

    STATUS=$(echo "$RESULT" | head -1 | cut -d'|' -f1)

    if [ "$STATUS" = "SKIP" ]; then
        REASON=$(echo "$RESULT" | head -1 | cut -d'|' -f2)
        echo ""
        echo "── $NODE: [SKIPPED] — $REASON ────────────────────────"
        echo "  PID    %CPU  %MEM  ELAPSED   CMD"
        echo "$RESULT" | grep "^JOBLINE|" | sed 's/^JOBLINE|//' | while read -r jobline; do
            echo "  $jobline"
        done
        SKIPPED+=("$NODE")
        continue
    fi

    if [ "$STATUS" = "ERROR" ]; then
        echo ""
        echo "── $NODE: [ERROR] — $(echo "$RESULT" | cut -d'|' -f2) ─────"
        continue
    fi

    AVG_UTIL=$(echo "$RESULT" | cut -d'|' -f2)
    TOTAL_USED=$(echo "$RESULT" | cut -d'|' -f3)
    TOTAL_MEM=$(echo "$RESULT" | cut -d'|' -f4)
    N_GPU=$(echo "$RESULT" | cut -d'|' -f5)
    LOAD1=$(echo "$RESULT" | cut -d'|' -f6)
    N_CPU=$(echo "$RESULT" | cut -d'|' -f7)
    OTHER_MEM=$(echo "$RESULT" | cut -d'|' -f8)

    # Composite load score (lower = better / more free).
    # Weighted: GPU utilization matters most, then GPU mem used, then CPU load ratio.
    CPU_LOAD_RATIO=$(awk "BEGIN {printf \"%.2f\", ($LOAD1/$N_CPU)*100}")
    SCORE_VAL=$(awk "BEGIN {printf \"%.2f\", ($AVG_UTIL*1.0) + ($OTHER_MEM/$TOTAL_MEM*100*0.5) + ($CPU_LOAD_RATIO*0.3)}")

    SCORE["$NODE"]=$SCORE_VAL
    SUMMARY["$NODE"]="GPUs=$N_GPU  avg_util=${AVG_UTIL}%  gpu_mem_used=${OTHER_MEM}MiB/${TOTAL_MEM}MiB  cpu_load(1m)=${LOAD1} (${N_CPU} cores)"
    USABLE+=("$NODE")

    echo ""
    echo "── $NODE: [OK] ───────────────────────────────────────"
    echo "  ${SUMMARY[$NODE]}"
    echo "  load score: $SCORE_VAL  (lower = more free)"
done

echo ""
echo "================================================================"
echo " RANKED ORDER — best (most free) to worst"
echo "================================================================"

if [ ${#USABLE[@]} -eq 0 ]; then
    echo "  No usable nodes found (all skipped, unreachable, or errored)."
else
    for NODE in "${USABLE[@]}"; do
        echo "${SCORE[$NODE]} $NODE"
    done | sort -n | awk '{printf "  %2d. %-15s score=%s\n", NR, $2, $1}'
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (you already have jobs running there):"
    printf '  %s\n' "${SKIPPED[@]}"
fi