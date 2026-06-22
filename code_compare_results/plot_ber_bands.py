"""
plot_ber_bands.py
==================
Reads the per-network BER evaluation results saved by evaluate_all_networks.py
(eval_stage_<s>.pt files, each {"SNR": tensor[N_snr], "BER": tensor[N_channels, N_snr]})
and plots BER vs SNR with a median line and shaded 25-75 percentile band for:

  - "Main" models: stage `MAIN_STAGE` of compare_networks_0..N-1
  - "Random" models: stage_3 of every sub-model under
    compare_networks_*/random_selections/*/

For each SNR point, BER values from ALL channels of ALL networks (in that
group) are pooled together, then the median and 25th/75th percentiles of that
pooled distribution are plotted.

Set MAIN_STAGE below to choose which stage represents the "main" models.
"""

import os
import torch
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_NAME = "compare_networks"
N_NETWORKS    = 100

BASE_PATH = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks"   # prefix where the compare_networks_x folders live
OUTPUT_DIR = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks/compare_networks_results"  # prefix where the resulting plot is saved

MAIN_STAGES  = [1, 3]  # which stage(s)' eval_stage_<s>.pt represent the "main" models
                           # (list with one element -> single band, as before)
RANDOM_STAGE = 3   # random_selections sub-models only have stage_3 by construction

SAVE_PATH = os.path.join(OUTPUT_DIR, "ber_vs_snr_bands.png")


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_eval(path: str, stage: int):
    """Load eval_stage_<stage>.pt from <path>/outputs/. Returns (SNR, BER) or None."""
    file_path = os.path.join(path, "outputs", f"eval_stage_{stage}.pt")
    if not os.path.exists(file_path):
        return None
    data = torch.load(file_path, weights_only=True)
    return data["SNR"], data["BER"]


def load_benchmark(main_path: str, bench_name: str):
    """Load worst_BER.pt + SNR.pt from a benchmark subfolder.
    bench_name is e.g. 'Single relay per channel' or 'Direct link'.
    Returns (SNR [S], BER [1, S]) or None."""
    folder = os.path.join(main_path, bench_name, bench_name, "outputs")
    snr_path = os.path.join(folder, "SNR.pt")
    ber_path = os.path.join(folder, "worst_BER.pt")
    if not os.path.exists(snr_path) or not os.path.exists(ber_path):
        return None
    snr = torch.load(snr_path, weights_only=True)
    ber = torch.load(ber_path, weights_only=True)   # [S]
    return snr, ber.unsqueeze(0)                     # [1, S]


def collect_benchmark(bench_name: str):
    """Pool worst_BER [1, S] from compare_networks_0..N-1 for a named benchmark."""
    snr_ref = None
    pooled  = []

    for idx in range(N_NETWORKS):
        main_path = os.path.join(BASE_PATH, f"{ORIGINAL_NAME}_{idx}", "")
        result = load_benchmark(main_path, bench_name)
        if result is None:
            print(f"  [{bench_name}] {ORIGINAL_NAME}_{idx}: not found, skipping")
            continue

        snr, ber = result
        if snr_ref is None:
            snr_ref = snr
        elif not torch.equal(snr, snr_ref):
            print(f"  [{bench_name}] {ORIGINAL_NAME}_{idx}: SNR grid mismatch, skipping")
            continue

        pooled.append(ber)

    return snr_ref, pooled


def collect_main_models(stage: int):
    """Pool BER values [C, N_snr] from compare_networks_0..N-1 (main model)."""
    snr_ref = None
    pooled = []  # list of tensors [C_i, N_snr]

    for idx in range(N_NETWORKS):
        main_path = os.path.join(BASE_PATH, f"{ORIGINAL_NAME}_{idx}", "")
        result = load_eval(main_path, stage)
        if result is None:
            print(f"  [main] {ORIGINAL_NAME}_{idx}: eval_stage_{stage}.pt not found, skipping")
            continue

        snr, ber = result
        if snr_ref is None:
            snr_ref = snr
        elif not torch.equal(snr, snr_ref):
            print(f"  [main] {ORIGINAL_NAME}_{idx}: SNR grid mismatch, skipping")
            continue

        pooled.append(ber.max(dim=0, keepdim=True).values)

    return snr_ref, pooled


def collect_random_models(stage: int):
    """Pool BER values [C, N_snr] from every random_selections sub-model
    across all compare_networks_0..N-1."""
    snr_ref = None
    pooled = []  # list of tensors [C_i, N_snr]

    n_found = 0
    n_with_eval = 0

    for idx in range(N_NETWORKS):
        main_path = os.path.join(BASE_PATH, f"{ORIGINAL_NAME}_{idx}", "")
        random_sel_path = os.path.join(main_path, "random_selections")
        if not os.path.isdir(random_sel_path):
            continue

        for sub_name in sorted(os.listdir(random_sel_path)):
            sub_path = os.path.join(random_sel_path, sub_name, "")
            if not os.path.isdir(sub_path):
                continue
            n_found += 1

            result = load_eval(sub_path, stage)
            if result is None:
                continue
            n_with_eval += 1

            snr, ber = result
            if snr_ref is None:
                snr_ref = snr
            elif not torch.equal(snr, snr_ref):
                print(f"  [random] {ORIGINAL_NAME}_{idx}/{sub_name}: SNR grid mismatch, skipping")
                continue

            pooled.append(ber.max(dim=0, keepdim=True).values)

    print(f"  [random] found {n_found} sub-model folders, "
          f"{n_with_eval} have outputs/eval_stage_{stage}.pt")

    return snr_ref, pooled


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_band(pooled: list, n_snr: int):
    """Stack [C_i, N_snr] tensors -> [sum(C_i), N_snr], then compute median,
    25th and 75th percentile across the pooled (network x channel) axis,
    for each SNR point. Returns (median, p25, p75) each shape [N_snr].

    Median is used instead of mean because on a log BER axis the arithmetic
    mean can be pulled far outside the 25-75% band by a few large outliers,
    whereas the median is guaranteed to lie between p25 and p75."""
    if not pooled:
        return None

    stacked = torch.cat(pooled, dim=0)  # [N_total, N_snr]

    median = torch.quantile(stacked, 0.50, dim=0)
    p25  = torch.quantile(stacked, 0.25, dim=0)
    p75  = torch.quantile(stacked, 0.75, dim=0)

    return median, p25, p75


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

MAIN_COLORS      = ['tab:blue', 'tab:green', 'tab:purple', 'tab:brown']
BENCHMARK_COLORS = {'Single relay per channel': 'tab:red',
                    'Direct link':              'tab:gray'}


def _masked_band(p25: torch.Tensor, p75: torch.Tensor):
    """Return (p25, p75) as numpy arrays with points where p75 == 0
    replaced by NaN so fill_between leaves a gap instead of a flat floor line."""
    p25_np = p25.numpy().copy()
    p75_np = p75.numpy().copy()
    zero_mask = p75_np <= 0
    p25_np[zero_mask] = float('nan')
    p75_np[zero_mask] = float('nan')
    return p25_np, p75_np


def plot_bands(main_entries, snr_random, band_random, bench_entries, save_path):
    """
    main_entries:  list of (stage, snr, band)
    bench_entries: list of (name, snr, band)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_yscale('log')

    for i, (stage, snr, band) in enumerate(main_entries):
        if band is None:
            continue
        _, p25, p75 = band
        snr_np = snr.numpy()
        color = MAIN_COLORS[i % len(MAIN_COLORS)]
        p25_np, p75_np = _masked_band(p25, p75)
        ax.fill_between(snr_np, p25_np, p75_np,
                         color=color, alpha=0.30,
                         label=f"Main models (stage {stage}) - 25-75%")

    if band_random is not None:
        _, p25, p75 = band_random
        snr_np = snr_random.numpy()
        p25_np, p75_np = _masked_band(p25, p75)
        ax.fill_between(snr_np, p25_np, p75_np,
                         color='tab:orange', alpha=0.35,
                         label=f"Random selections (stage {RANDOM_STAGE}) - 25-75%")

    for name, snr, band in bench_entries:
        if band is None:
            continue
        _, p25, p75 = band
        snr_np = snr.numpy()
        color  = BENCHMARK_COLORS.get(name, 'black')
        p25_np, p75_np = _masked_band(p25, p75)
        ax.fill_between(snr_np, p25_np, p75_np,
                         color=color, alpha=0.25,
                         label=f"{name} - 25-75%")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("BER (log scale)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', ncol=1, fontsize=9)
    ax.set_ylim(bottom=1e-5, top=1)
    # ax.set_xlim(left=-10, right=0)  # adjust as needed based on your BER range
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    main_entries = []  # list of (stage, snr, band)
    for stage in MAIN_STAGES:
        print(f"Collecting main models (stage {stage})...")
        snr_main, pooled_main = collect_main_models(stage)
        band_main = compute_band(pooled_main, n_snr=len(snr_main)) if snr_main is not None else None
        if pooled_main:
            print(f"  pooled {len(pooled_main)} worst-channel curves")
        main_entries.append((stage, snr_main, band_main))

    print(f"\nCollecting random selections (stage {RANDOM_STAGE})...")
    snr_random, pooled_random = collect_random_models(RANDOM_STAGE)
    band_random = compute_band(pooled_random, n_snr=len(snr_random)) if snr_random is not None else None
    if pooled_random:
        print(f"  pooled {len(pooled_random)} worst-channel curves")

    bench_entries = []
    for bench_name in ("Single relay per channel", "Direct link"):
        print(f"\nCollecting benchmark: {bench_name}...")
        snr_bench, pooled_bench = collect_benchmark(bench_name)
        band_bench = compute_band(pooled_bench, n_snr=len(snr_bench)) if snr_bench is not None else None
        if pooled_bench:
            print(f"  pooled {len(pooled_bench)} curves")
        bench_entries.append((bench_name, snr_bench, band_bench))

    if all(b is None for _, _, b in main_entries) and band_random is None \
            and all(b is None for _, _, b in bench_entries):
        print("No data found - nothing to plot.")
        return

    plot_bands(main_entries, snr_random, band_random, bench_entries, SAVE_PATH)


if __name__ == "__main__":
    main()