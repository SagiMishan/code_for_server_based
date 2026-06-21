import os
import re
from collections import defaultdict


def seconds_to_hhmmss(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    # format guarantees 2 digits for hours/minutes, and pads seconds with decimals
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


DEFAULT_DIR = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks"
DEFAULT_NAME = "compare_networks"
CLAIM_DIR = "/home/dsi/mishans1/projects/code_for_server_based/simulation/claims"
NUBER_GPU = 6
stage_1_time = 0
stage_2_time = 0
stage_3_time = 0
total_time = 0
counter = 0
open_and_not_completed = []
time_pattern = r"\d+:\d+:\d+\.\d+"
completed_networks = []

# ── per-server stats ──────────────────────────────────────────────────────────
# server -> dict with counter, stage_1_time, stage_2_time, stage_3_time, total_time
server_stats = defaultdict(lambda: {
    "counter": 0,
    "stage_1_time": 0.0,
    "stage_2_time": 0.0,
    "stage_3_time": 0.0,
    "total_time": 0.0,
})


def get_owner_server(idx):
    """Look up which server trained network idx via its claim folder's
    owner.txt (format: '<node> GPU<id> <date...>'). Returns None if no
    claim record exists (e.g. claims were cleared after completion)."""
    owner_path = os.path.join(CLAIM_DIR, f"task_{idx}.claim", "owner.txt")
    if not os.path.isfile(owner_path):
        return None
    with open(owner_path, "r") as f:
        content = f.read().strip()
    if not content:
        return None
    return content.split()[0]  # first token is the node name


for dir in os.listdir(DEFAULT_DIR):
    if dir.startswith(DEFAULT_NAME):
        if os.path.isfile(os.path.join(DEFAULT_DIR, dir, "timing.txt")):
            counter += 1
            idx = int(dir.split("_")[-1])
            completed_networks.append(idx)

            with open(os.path.join(DEFAULT_DIR, dir, "timing.txt"), "r") as f:
                content = f.read()
                times = re.findall(time_pattern, content)
                if times:
                    t1 = float(times[0].split(':')[2]) + float(times[0].split(':')[1]) * 60 + float(times[0].split(':')[0]) * 3600
                    t2 = float(times[1].split(':')[2]) + float(times[1].split(':')[1]) * 60 + float(times[1].split(':')[0]) * 3600
                    t3 = float(times[2].split(':')[2]) + float(times[2].split(':')[1]) * 60 + float(times[2].split(':')[0]) * 3600

                    stage_1_time += t1
                    stage_2_time += t2
                    stage_3_time += t3

                    server = get_owner_server(idx)
                    if server is not None:
                        s = server_stats[server]
                        s["counter"] += 1
                        s["stage_1_time"] += t1
                        s["stage_2_time"] += t2
                        s["stage_3_time"] += t3
                        s["total_time"] += (t1 + t2 + t3)
        else:
            if dir.split("_")[-1].isdigit():
                open_and_not_completed.append(int(dir.split("_")[-1]))

total_time += stage_1_time + stage_2_time + stage_3_time

print(f"Total {counter} tests completed.")
print(f"Open and not completed: {len(open_and_not_completed)}")
print(f"Average for Stage 1 Time: {seconds_to_hhmmss(stage_1_time / counter if counter > 0 else 0)}")
print(f"Average for Stage 2 Time: {seconds_to_hhmmss(stage_2_time / counter if counter > 0 else 0)}")
print(f"Average for Stage 3 Time: {seconds_to_hhmmss(stage_3_time / counter if counter > 0 else 0)}")
print(f"Average for Total Time:   {seconds_to_hhmmss(total_time / counter if counter > 0 else 0)}")
time_left = (100 - counter) * (total_time / counter if counter > 0 else 0)
time_left = time_left / (NUBER_GPU * 4)
print(f"Estimated time left for remaining tests: {seconds_to_hhmmss(time_left)}")
print(f"Completed networks: {sorted(completed_networks)}")
print(f"Open and not completed: {sorted(open_and_not_completed)}")

# ── per-server statistics ───────────────────────────────────────────────────
print("")
print("=" * 60)
print(" PER-SERVER STATISTICS")
print("=" * 60)

if not server_stats:
    print("  No server ownership data found "
          "(claim folders may have been cleared).")
else:
    for server in sorted(server_stats.keys()):
        s = server_stats[server]
        n = s["counter"]
        avg_total = s["total_time"] / n if n > 0 else 0
        print(f"\n  {server}:")
        print(f"    Networks completed:      {n}")
        print(f"    Avg Stage 1 Time:        {seconds_to_hhmmss(s['stage_1_time'] / n if n > 0 else 0)}")
        print(f"    Avg Stage 2 Time:        {seconds_to_hhmmss(s['stage_2_time'] / n if n > 0 else 0)}")
        print(f"    Avg Stage 3 Time:        {seconds_to_hhmmss(s['stage_3_time'] / n if n > 0 else 0)}")
        print(f"    Avg Total Time/network:  {seconds_to_hhmmss(avg_total)}")

    # quick comparison: which server is fastest on average
    fastest = min(server_stats.items(), key=lambda kv: (kv[1]["total_time"] / kv[1]["counter"]) if kv[1]["counter"] > 0 else float("inf"))
    print(f"\n  Fastest server (avg time/network): {fastest[0]} "
          f"({seconds_to_hhmmss(fastest[1]['total_time'] / fastest[1]['counter'])})")