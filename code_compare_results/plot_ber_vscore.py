"""
plot_ber_vscore.py
==================
Plots BER_log_snr and V score on a shared x-axis with two y-axes.

Usage
-----
    python plot_ber_vscore.py                    # default range 0-9
    python plot_ber_vscore.py --start 0 --end 4  # average over networks 0..4
    python plot_ber_vscore.py --start 5 --end 5  # single network 5
    python plot_ber_vscore.py --stage 2          # choose which stage log to use

Arguments
---------
    --start   first network index  (default 0)
    --end     last  network index  (default 9)
    --stage   1, 2, or 3          (default 3)
    --base    path to compare_networks folder
    --out     output PNG filename
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

DEFAULT_BASE = "simulation/compare_networks"
MODEL_NAME   = "compare_networks"


def load_stage_log(base, idx, stage):
    model_path = os.path.join(base, f"{MODEL_NAME}_{idx}")
    if stage == 2:
        fpath = os.path.join(model_path, "outputs", "stage2_all_epochs_raw.pt")
    else:
        fpath = os.path.join(model_path, "outputs", f"stage{stage}_log.pt")

    if not os.path.isfile(fpath):
        print(f"  [skip] {fpath} not found")
        return None
    try:
        return torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [skip] {fpath} — {e}")
        return None


def extract_series(log):
    ber_raw    = log.get("BER_log_snr", [])
    vscore_raw = log.get("v_score", [])
    n = min(len(ber_raw), len(vscore_raw))
    if n == 0:
        return None, None, None
    ber    = np.array([float(b) if b is not None else np.nan for b in ber_raw[:n]])
    vscore = np.array([float(v) for v in vscore_raw[:n]])
    x      = np.arange(n)
    return x, ber, vscore


def _dual_axes(fig, ax_ber, x, ber, vscore, ber_mean=None, v_mean=None,
               ber_std=None, v_std=None, snr_log=None, is_avg=False):
    """Draw BER (left, log) and V score (right, linear) on ax_ber."""
    ber_label = f"BER @ {snr_log:.0f} dB" if snr_log is not None else "BER (log SNR)"

    # ── BER ───────────────────────────────────────────────────────────────────
    valid = ~np.isnan(ber)
    ax_ber.semilogy(x[valid], ber[valid], color="crimson", linewidth=2,
                    alpha=0.85, label=ber_label, zorder=3)
    if is_avg and ber_std is not None:
        lo = np.clip(ber - ber_std, 1e-7, None)
        hi = ber + ber_std
        ax_ber.fill_between(x[valid], lo[valid], hi[valid],
                            color="crimson", alpha=0.15, zorder=2)
    ax_ber.set_xlabel("Training step (×100 iterations)", fontsize=11)
    ax_ber.set_ylabel("BER (log scale)", color="crimson", fontsize=11)
    ax_ber.tick_params(axis="y", labelcolor="crimson")
    ax_ber.grid(True, which="both", linestyle="--", alpha=0.3)

    # ── V score ───────────────────────────────────────────────────────────────
    ax_v = ax_ber.twinx()
    ax_v.plot(x, vscore, color="steelblue", linewidth=2,
              alpha=0.85, label="V score", zorder=3)
    if is_avg and v_std is not None:
        ax_v.fill_between(x,
                          np.clip(vscore - v_std, 0, 1),
                          np.clip(vscore + v_std, 0, 1),
                          color="steelblue", alpha=0.15, zorder=2)
    ax_v.set_ylabel("V score (total)", color="steelblue", fontsize=11)
    ax_v.tick_params(axis="y", labelcolor="steelblue")
    ax_v.set_ylim(-0.05, 1.05)

    # ── Combined legend ───────────────────────────────────────────────────────
    h1, l1 = ax_ber.get_legend_handles_labels()
    h2, l2 = ax_v.get_legend_handles_labels()
    ax_ber.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    return ax_v


def plot_single(x, ber, vscore, title, snr_log, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    _dual_axes(fig, ax, x, ber, vscore, snr_log=snr_log)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_average(all_x, all_ber, all_vscore, title, snr_log, out_path):
    n = min(len(x) for x in all_x)
    ber_mat    = np.stack([b[:n] for b in all_ber],    axis=0)
    vscore_mat = np.stack([v[:n] for v in all_vscore], axis=0)
    x = np.arange(n)

    ber_mean = np.nanmean(ber_mat,    axis=0)
    ber_std  = np.nanstd(ber_mat,     axis=0)
    v_mean   = np.mean(vscore_mat,    axis=0)
    v_std    = np.std(vscore_mat,     axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    _dual_axes(fig, ax, x, ber_mean, v_mean,
               ber_std=ber_std, v_std=v_std,
               snr_log=snr_log, is_avg=True)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=9)
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--base",  type=str, default=DEFAULT_BASE)
    parser.add_argument("--out",   type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or f"ber_vscore_stage{args.stage}_{args.start}-{args.end}.png"
    indices  = list(range(args.start, args.end + 1))
    print(f"Loading stage {args.stage} logs for networks {args.start}..{args.end}")

    all_x, all_ber, all_vscore = [], [], []
    snr_log_val = None

    for idx in indices:
        log = load_stage_log(args.base, idx, args.stage)
        if log is None:
            continue
        if snr_log_val is None:
            snr_log_val = log.get("SNR_log")
        x, ber, vscore = extract_series(log)
        if x is None:
            print(f"  [skip] network {idx} — no usable data")
            continue
        all_x.append(x)
        all_ber.append(ber)
        all_vscore.append(vscore)
        print(f"  network {idx}: {len(x)} steps")

    if not all_x:
        print("No data found.")
        sys.exit(1)

    n_loaded = len(all_x)

    if n_loaded == 1:
        title = f"Stage {args.stage} — network {args.start}"
        plot_single(all_x[0], all_ber[0], all_vscore[0],
                    title, snr_log_val, out_path)
    else:
        title = (f"Stage {args.stage} — average over {n_loaded} networks "
                 f"(idx {args.start}–{args.end})   shaded = ±1 std")
        plot_average(all_x, all_ber, all_vscore,
                     title, snr_log_val, out_path)


if __name__ == "__main__":
    main()
