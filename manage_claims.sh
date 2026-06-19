#!/bin/bash
# Helper for inspecting / resetting task claims.
#
# Usage:
#   bash manage_claims.sh status              # show claimed/done/pending tasks
#   bash manage_claims.sh clear <task_id>      # remove claim for one task (allows re-run)
#   bash manage_claims.sh clear-all            # remove ALL claims (allows full re-run)

CLAIM_DIR=simulation/claims

case "$1" in
    status)
        echo "Claimed tasks:"
        for d in "$CLAIM_DIR"/task_*; do
            [ -d "$d" ] || continue
            TASK_ID=$(basename "$d" | sed 's/task_//')
            OWNER=$(cat "$d/owner.txt" 2>/dev/null || echo "unknown")
            if [ -f "$d/done.txt" ]; then
                STATUS="DONE"
            else
                STATUS="RUNNING"
            fi
            printf "  task %-4s [%s]  owner: %s\n" "$TASK_ID" "$STATUS" "$OWNER"
        done
        ;;
    clear)
        if [ -z "$2" ]; then
            echo "Usage: bash manage_claims.sh clear <task_id>"
            exit 1
        fi
        rm -rf "$CLAIM_DIR/task_$2"
        echo "Cleared claim for task $2 (it can now be re-run)"
        ;;
    clear-all)
        read -p "This will clear ALL claims and allow every task to be re-run. Continue? [y/N] " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf "$CLAIM_DIR"
            mkdir -p "$CLAIM_DIR"
            echo "All claims cleared."
        else
            echo "Cancelled."
        fi
        ;;
    *)
        echo "Usage:"
        echo "  bash manage_claims.sh status"
        echo "  bash manage_claims.sh clear <task_id>"
        echo "  bash manage_claims.sh clear-all"
        ;;
esac
