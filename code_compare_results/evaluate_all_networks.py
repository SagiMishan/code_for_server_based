"""
evaluate_all_networks.py
=========================
For every main model "compare_networks_<x>" (x = 0..N_NETWORKS-1):

  1. Evaluate the main model at every available stage (stage_1 / stage_2 / stage_3)
     -> save SNR & worst-BER tensors to outputs/eval_stage_<s>.pt

  2. Evaluate every sub-model in <main>/random_selections/<name>/ at every
     available stage (only stage_3 exists for these by construction, but the
     script checks generically) -> save to
     <main>/random_selections/<name>/outputs/eval_stage_<s>.pt

Each saved .pt file is a dict: {"SNR": tensor[N_snr], "BER": tensor[N_channels, N_snr]}

Built on top of Network_multy_channels.load_model / evaluate_model
(same conventions as compare_models.py).

── PARALLEL EXECUTION ─────────────────────────────────────────────────────────
This script trains/evaluates one realisation (idx) at a time and is safe to
run across multiple GPUs and multiple nodes simultaneously:

  - Set CUDA_VISIBLE_DEVICES=<gpu_id>            to pin a process to one GPU
  - Set SLURM_ARRAY_TASK_ID=<idx>                to evaluate only that one idx
  - Without SLURM_ARRAY_TASK_ID, all idx 0..N_NETWORKS-1 run sequentially
    (original behaviour).

Cross-node / cross-GPU safety is handled with the same atomic `mkdir` claim
mechanism used by run_all_gpu_safe.sh: before evaluating model idx, a process
atomically creates simulation/eval_claims/eval_<idx>.claim. mkdir is atomic
even across machines sharing the same NFS mount, so two GPUs (on the same or
different nodes) can never evaluate the same network twice.

Use run_eval_all_gpu_safe.sh to launch this across all your idle GPUs/nodes.
"""

import sys
import os
from pathlib import Path
# Add the main folder (parent of code_compare_results, etc.) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from code_Networks.Network_multy_channels import Network_multy_channel, load_model, dB2lin


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH     = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks"
ORIGINAL_NAME = "compare_networks"
N_NETWORKS    = 10

SNR_RANGE = torch.linspace(-20, 40, 121)  # SNR points (dB) to evaluate
BATCH     = 10 ** 6
NUM_ITR   = 1

STAGES = (1, 2, 3)

# If True, skip evaluation for any (model, stage) that already has
# outputs/eval_stage_<s>.pt saved on disk.
SKIP_EXISTING = True

# Claim folder for cross-node/cross-GPU coordination (NFS-shared)
CLAIM_DIR = "/home/dsi/mishans1/projects/code_for_server_based/simulation/eval_claims"


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def _select_device() -> str:
    """Pick cuda/mps/cpu, but verify CUDA actually runs a kernel
    (torch.cuda.is_available() can be True even when no compatible
    kernel image exists for the installed GPU)."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda") + 1
            return "cuda"
        except RuntimeError as e:
            print(f"CUDA reported available but unusable ({e}); falling back to CPU")
            return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Claim helpers (cross-node safe, NFS-backed, atomic via mkdir)
# ─────────────────────────────────────────────────────────────────────────────

def try_claim(idx: int) -> bool:
    """Atomically claim evaluation of network idx. Returns True if this
    process won the claim, False if another process already owns it."""
    os.makedirs(CLAIM_DIR, exist_ok=True)
    claim_path = os.path.join(CLAIM_DIR, f"eval_{idx}.claim")
    try:
        os.mkdir(claim_path)  # atomic on POSIX/NFS - fails if it already exists
    except FileExistsError:
        return False

    node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename
    gpu  = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(os.path.join(claim_path, "owner.txt"), "w") as f:
        f.write(f"{node} GPU{gpu}\n")
    return True


def mark_done(idx: int) -> None:
    claim_path = os.path.join(CLAIM_DIR, f"eval_{idx}.claim")
    open(os.path.join(claim_path, "done.txt"), "w").close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def stage_files_exist(path: str, stage: int, n_channels: int) -> bool:
    """Check that models/python/stage_<s>c<c> exists for every channel."""
    python_models_path = os.path.join(path, "models", "python")
    return all(
        os.path.exists(os.path.join(python_models_path, f"stage_{stage}c{c}"))
        for c in range(n_channels)
    )


def eval_result_exists(outputs_dir: str, stage: int) -> bool:
    """Check if outputs/eval_stage_<s>.pt already exists."""
    return os.path.exists(os.path.join(outputs_dir, f"eval_stage_{stage}.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: Network_multy_channel,
                   snr_range: torch.Tensor,
                   batch: int,
                   num_itr: int) -> torch.Tensor:
    """Return worst-case BER tensor of shape [N_channels, len(snr_range)]."""
    C   = model.N_channels
    out = torch.zeros(C, len(snr_range))
    model.eval()
    with torch.no_grad():
        for c in range(C):
            for si, snr in enumerate(snr_range):
                model.sub_networks[c].SNR = dB2lin(snr).to(model.sub_networks[c]._device)
                acc = 0.0
                for _ in range(num_itr):
                    signal, bits = model.sub_networks[c].modulator(batch)
                    rm   = model.sub_networks[c](signal, bits)
                    pred = model.sub_networks[c].demodulator(rm)
                    worst, _, _ = model.sub_networks[c].BER(bits=bits, pred=pred)
                    acc += float(worst)
                out[c, si] = acc / num_itr
                print(f"    ch{c} SNR={float(snr):5.1f} dB  BER={out[c, si]:.3e}")
                if acc == 0:
                    break  # BER is 0 and monotone non-increasing; skip higher SNRs
    return out


def evaluate_and_save_stages(data_path: str,
                              weight_path: str,
                              label: str) -> None:
    """Load model architecture from data_path (where data/ lives),
    load weights from weight_path (where models/python/ lives),
    evaluate each available stage and save results to weight_path/outputs/.

    For the main model:       data_path == weight_path == main_path
    For random sub-models:    data_path  = main_path (has data/)
                              weight_path = sub_path  (has models/python/ + outputs/)
    """
    outputs_dir = os.path.join(weight_path, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Build model once from main data/, move to device — weights loaded per stage below
    model = load_model(data_path, stage=None)
    model.to(DEVICE)

    for stage in STAGES:
        if SKIP_EXISTING and eval_result_exists(outputs_dir, stage):
            print(f"  [{label}] stage_{stage}: result exists, skipping")
            continue

        if not stage_files_exist(weight_path, stage, model.N_channels):
            print(f"  [{label}] stage_{stage}: weights not found, skipping")
            continue

        print(f"  [{label}] stage_{stage}: loading and evaluating on {DEVICE}")
        model.load(weight_path, f"stage_{stage}")
        ber = evaluate_model(model, SNR_RANGE, BATCH, NUM_ITR)

        save_path = os.path.join(outputs_dir, f"eval_stage_{stage}.pt")
        torch.save({"SNR": SNR_RANGE, "BER": ber.cpu()}, save_path)
        print(f"  [{label}] stage_{stage}: saved -> {save_path}")


def evaluate_one_network(idx: int) -> None:
    """Evaluate a single network realisation (all its sub-models)."""
    name_of_model = f"{ORIGINAL_NAME}_{idx}"
    main_path = os.path.join(BASE_PATH, name_of_model, "")

    if not os.path.isdir(main_path):
        print(f"{name_of_model}: folder not found, skipping")
        return

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        # ── main model ──────────────────────────────────────────────────
        evaluate_and_save_stages(data_path=main_path,
                                 weight_path=main_path,
                                 label="main")

        # ── random sub-models ───────────────────────────────────────────
        random_sel_path = os.path.join(main_path, "random_selections")
        if os.path.isdir(random_sel_path):
            for sub_name in sorted(os.listdir(random_sel_path)):
                sub_path = os.path.join(random_sel_path, sub_name, "")
                if os.path.isdir(sub_path):
                    evaluate_and_save_stages(data_path=main_path,
                                        weight_path=sub_path,
                                        label=f"random_selections/{sub_name}")
        # ── QAM sub-models ───────────────────────────────────────────
        qam_sel_path = os.path.join(main_path, "QAM baseline")
        if os.path.isdir(qam_sel_path):
            for sub_name in sorted(os.listdir(qam_sel_path)):
                sub_path = os.path.join(qam_sel_path, sub_name, "")
                if os.path.isdir(sub_path):
                    evaluate_and_save_stages(data_path=main_path,
                                        weight_path=sub_path,
                                     label=f"QAM baseline/{sub_name}")
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"{name_of_model}: done.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if array_id is not None:
        # Running as one realisation in a parallel job — claim-protected
        idx = int(array_id)
        if not try_claim(idx):
            print(f"compare_networks_{idx}: already claimed by another process, skipping")
            return
        evaluate_one_network(idx)
        mark_done(idx)
    else:
        # Original sequential behaviour, also claim-protected so it's safe
        # to run alongside parallel workers without duplicating work.
        for idx in tqdm(range(N_NETWORKS), desc="Evaluating networks"):
            if not try_claim(idx):
                print(f"compare_networks_{idx}: already claimed by another process, skipping")
                continue
            evaluate_one_network(idx)
            mark_done(idx)


if __name__ == "__main__":
    main()