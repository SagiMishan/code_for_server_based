#!/bin/bash
# Helper for inspecting / resetting random-selection claims.
#
# Claims are organized as: simulation/random_claims/<idx>/<hex_name>.claim
#
# Usage:
#   bash manage_random_claims.sh status [idx]        # show claims (all idx, or just one)
#   bash manage_random_claims.sh clear <idx> <name>    # remove claim for one selection
#   bash manage_random_claims.sh clear-idx <idx>        # remove ALL claims for one idx
#   bash manage_random_claims.sh clear-all              # remove ALL claims (every idx)

CLAIM_DIR=/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_claims

print_idx_status () {
    local idx=$1
    local idx_dir="$CLAIM_DIR/$idx"
    [ -d "$idx_dir" ] || { echo "  (no claims yet for idx $idx)"; return; }

    local n_done=0
    local n_running=0
    for d in "$idx_dir"/*.claim; do
        [ -d "$d" ] || continue
        local name=$(basename "$d" .claim)
        local owner=$(cat "$d/owner.txt" 2>/dev/null || echo "unknown")
        if [ -f "$d/done.txt" ]; then
            status="DONE"
            n_done=$((n_done + 1))
        else
            status="RUNNING"
            n_running=$((n_running + 1))
        fi
        printf "    %-12s [%s]  owner: %s\n" "$name" "$status" "$owner"
    done
    echo "  idx $idx summary: $n_done done, $n_running running"
}

case "$1" in
    status)
        if [ -n "$2" ]; then
            echo "Claims for idx $2:"
            print_idx_status "$2"
        else
            echo "Claims for all idx:"
            if [ ! -d "$CLAIM_DIR" ]; then
                echo "  (no claims directory yet)"
            else
                for idx_dir in "$CLAIM_DIR"/*/; do
                    [ -d "$idx_dir" ] || continue
                    idx=$(basename "$idx_dir")
                    echo "idx $idx:"
                    print_idx_status "$idx"
                done
            fi

            echo ""
            echo "Servers currently in use (RUNNING claims):"
            declare -a ACTIVE_NODES=()
            for d in "$CLAIM_DIR"/*/*.claim; do
                [ -d "$d" ] || continue
                [ -f "$d/done.txt" ] && continue
                OWNER=$(cat "$d/owner.txt" 2>/dev/null)
                NODE_PART=$(echo "$OWNER" | awk '{print $1}')
                [ -n "$NODE_PART" ] && ACTIVE_NODES+=("$NODE_PART")
            done
            if [ ${#ACTIVE_NODES[@]} -eq 0 ]; then
                echo "  (none — no selections currently RUNNING)"
            else
                printf '  %s\n' "${ACTIVE_NODES[@]}" | sort -u
            fi
        fi
        ;;
    clear)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: bash manage_random_claims.sh clear <idx> <name>"
            exit 1
        fi
        rm -rf "$CLAIM_DIR/$2/$3.claim"
        echo "Cleared claim for idx $2 / selection $3 (it can now be re-run)"
        ;;
    clear-idx)
        if [ -z "$2" ]; then
            echo "Usage: bash manage_random_claims.sh clear-idx <idx>"
            exit 1
        fi
        read -p "This will clear ALL claims for idx $2. Continue? [y/N] " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf "$CLAIM_DIR/$2"
            echo "Cleared all claims for idx $2."
        else
            echo "Cancelled."
        fi
        ;;
    clear-all)
        read -p "This will clear ALL claims for EVERY idx and allow full re-run. Continue? [y/N] " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf "$CLAIM_DIR"
            mkdir -p "$CLAIM_DIR"
            echo "All random-selection claims cleared."
        else
            echo "Cancelled."
        fi
        ;;
    *)
        echo "Usage:"
        echo "  bash manage_random_claims.sh status [idx]"
        echo "  bash manage_random_claims.sh clear <idx> <name>"
        echo "  bash manage_random_claims.sh clear-idx <idx>"
        echo "  bash manage_random_claims.sh clear-all"
        ;;
esac
