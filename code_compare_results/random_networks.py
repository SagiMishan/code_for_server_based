import time
from tqdm import tqdm
import sys
import os
from pathlib import Path
# Add the main folder (parent of code_compare_results, etc.) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_Networks.SmallFunctions import _fmt
original_stdout = sys.stdout

import matplotlib.pyplot as plt

from code_Networks.Network_multy_channels import Network_multy_channel, load_model
from code_Networks.Train_funcatios_clude import *
from code_Networks.SmallFunctions import *


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ── Paths ──────────────────────────────────────────────────────────────────────
# RANDOM_CLAIM_DIR is read from the environment variable BASE_CLAIM_DIR set by
# run_random_networks_gpu_safe.sh, so the bash script and python file always
# agree on the path. Falls back to a sensible default for manual runs.
RANDOM_CLAIM_DIR = os.environ.get(
    "BASE_CLAIM_DIR",
    "/home/dsi/mishans1/projects/code_for_server_based/simulation/log_files/compare_networks/random_claims"
)

MAIN_MODEL_BASE = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks"

orignal_Name_of_model = "compare_networks"
N_networks = 10
SNR_basic_trainning = 50
max_iteration = 5
SNR_step = 1
BER_th = 0.5e-3

# ── Per-process unique random seed ─────────────────────────────────────────────
# IMPORTANT: do NOT use a fixed seed here. A fixed seed makes every parallel
# process generate the EXACT SAME sequence of "random" draws, causing constant
# claim collisions and effectively serialising all workers onto the same
# selections. Seed uniquely per process using PID + time + node hash.
import hashlib
_node_hash = int(hashlib.md5(os.uname().nodename.encode()).hexdigest()[:8], 16)
_unique_seed = (os.getpid() * 2654435761 + time.time_ns() + _node_hash) % (2**31 - 1)
torch.manual_seed(_unique_seed)
print(f"[random_networks] Using per-process random seed: {_unique_seed}")


# ── Claim helpers ──────────────────────────────────────────────────────────────

def try_claim_selection(idx: int, name: str) -> bool:
    """Atomically claim training of random selection `name` under model idx.
    Returns True if this process won the claim, False if already claimed."""
    claim_root = os.path.join(RANDOM_CLAIM_DIR, str(idx))
    os.makedirs(claim_root, exist_ok=True)
    claim_path = os.path.join(claim_root, f"{name}.claim")
    try:
        os.mkdir(claim_path)  # atomic on POSIX/NFS
    except FileExistsError:
        return False

    node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
    gpu  = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(os.path.join(claim_path, "owner.txt"), "w") as f:
        f.write(f"{node} GPU{gpu}\n")
    return True


def mark_selection_done(idx: int, name: str) -> None:
    claim_path = os.path.join(RANDOM_CLAIM_DIR, str(idx), f"{name}.claim")
    open(os.path.join(claim_path, "done.txt"), "w").close()


def count_existing_random_selections(main_path, n_channels):
    """Count how many random_selections subfolders already contain a finished stage_3 model."""
    random_sel_path = os.path.join(main_path, "random_selections")
    if not os.path.exists(random_sel_path):
        return 0

    count = 0
    for name in os.listdir(random_sel_path):
        sub_path = os.path.join(random_sel_path, name)
        if not os.path.isdir(sub_path):
            continue
        python_models_path = os.path.join(sub_path, "models", "python")
        if all(os.path.exists(os.path.join(python_models_path, f"stage_3c{c}"))
               for c in range(n_channels)):
            count += 1
    return count


# ── Main logic ─────────────────────────────────────────────────────────────────

def process_one_idx(idx: int) -> None:
    Name_of_model = f"{orignal_Name_of_model}_{idx}"
    main_path = os.path.join(MAIN_MODEL_BASE, Name_of_model, "")

    model = load_model(path=main_path).to(device)
    max_snr_train = torch.load(
        os.path.join(main_path, "data", "max_snr_train_stage_1"), weights_only=True)

    random_sel_dir = os.path.join(main_path, "random_selections")
    if not os.path.exists(random_sel_dir):
        os.makedirs(random_sel_dir)

    n_existing = count_existing_random_selections(main_path, model.N_channels)
    n_to_run   = max(0, N_networks - n_existing)
    print(f"{Name_of_model}: found {n_existing} existing random selections, "
          f"running {n_to_run} more")

    completed_this_run = 0
    attempts   = 0
    collisions = 0
    max_attempts = n_to_run * 10 + 20  # safety cap against infinite retry loops

    pbar = tqdm(total=n_to_run, desc=Name_of_model)
    while completed_this_run < n_to_run and attempts < max_attempts:
        attempts += 1
        start_time_1 = time.time()

        model = load_model(path=main_path).to(device)
        for c in range(model.N_channels):
            model.P[c] = torch.rand(model.P[c].shape)

        model.set_weights()

        p_r = torch.stack([model.P[c].detach().squeeze()
                           for c in range(model.N_channels)])  # [C, N]
        winner = torch.zeros_like(p_r)
        winner[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = 1.0

        name_of_model_random = winner[0, :].to(dtype=torch.int)
        binary_str = ''.join(map(str, name_of_model_random.flatten().tolist()))
        padded_binary_str = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
        name_of_model_random = ''.join(
            f'{int(padded_binary_str[i:i + 4], 2):X}'
            for i in range(0, len(padded_binary_str), 4))

        path = os.path.join(main_path, "random_selections", name_of_model_random, "")

        # Skip if already finished on disk
        python_models_path = os.path.join(path, "models", "python")
        if all(os.path.exists(os.path.join(python_models_path, f"stage_3c{c}"))
               for c in range(model.N_channels)):
            print(f"Skipping {name_of_model_random}, already exists on disk")
            continue

        # Atomic claim — prevents two GPUs from training the same selection
        if not try_claim_selection(idx, name_of_model_random):
            collisions += 1
            print(f"Skipping {name_of_model_random}, already claimed by another process "
                  f"(collisions={collisions}/{attempts} so far)")
            continue

        print(f"Selected model in {orignal_Name_of_model}-{idx}: {name_of_model_random}")

        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, "models"))
            os.makedirs(os.path.join(path, "models", "python"))
            os.makedirs(os.path.join(path, "models", "matlab"))
            os.makedirs(os.path.join(path, "outputs"))
            os.makedirs(os.path.join(path, "data"))
        # sys.stdout = open(os.devnull, 'w')
        try:
            stage_3(model=model,
                    SNR_basic_trainning=SNR_basic_trainning,
                    SNR_max=max_snr_train + 5,
                    BER_th=BER_th,
                    device=device,
                    SNR_step=SNR_step,
                    max_iteration=max_iteration,
                    path=path,
                    SNR_val=0)
            plot_architecture(
                path=main_path,
                Name_of_model=os.path.join("random_selections", name_of_model_random, ""),
                stage="stage_3")
            plt.close()
        finally:
            # sys.stdout.close()
            # sys.stdout = original_stdout
            end_time_1 = time.time()
            timing_lines = [f"stage_3 took {_fmt(end_time_1 - start_time_1)} (hh:mm:ss)"]
            for line in timing_lines:
                print(line)

        with open(os.path.join(path, "timing.txt"), "w") as f:
            f.write("\n".join(timing_lines) + "\n")

        mark_selection_done(idx, name_of_model_random)
        completed_this_run += 1
        pbar.update(1)

    pbar.close()
    collision_rate = (collisions / attempts * 100) if attempts > 0 else 0
    print(f"{Name_of_model}: finished — {completed_this_run}/{n_to_run} completed, "
          f"{attempts} total attempts, {collisions} collisions "
          f"({collision_rate:.1f}% wasted on collisions)")
    if attempts >= max_attempts:
        print(f"{Name_of_model}: hit max_attempts ({max_attempts}) — "
              f"completed {completed_this_run}/{n_to_run}.")


def main():
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if array_id is not None:
        process_one_idx(int(array_id))
    else:
        for idx in range(10):
            process_one_idx(idx)


if __name__ == "__main__":
    main()