"""
plot_saved_ber.py
=================
Plots worst BER from pre-computed eval_stage_<s>.pt files.
No model loading / simulation needed.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_PATH     = "/home/dsi/mishans1/projects/code_for_server_based/"
ORIGINAL_NAME = "multy_anntena"
MODEL_INDICES = list(range(0, 10))

STAGES        = [3]

QAM_SUB_ROOT  = "QAM baseline"   # set None to skip simple-demod curves

OUT_DIR  = "/home/dsi/mishans1/projects/code_for_server_based/compare_networks_results/"
OUT_FILE = "ber_multy_anntena.png"

MIN_BER  = 1e-5   # y-axis lower limit

# x-axis limits — set to None for auto
SNR_XLIM = (-10 ,0)         # e.g. (-5, 30)
# ══════════════════════════════════════════════════════════════════════════════

STAGE_LINESTYLE = {1: "-",  2: "--", 3: "-."}
STAGE_MARKER    = {1: "o",  2: "s",  3: "^"}
STAGE_DASHES    = {1: (6, 0), 2: (5, 2), 3: (5, 2, 1, 2)}
SIMPLE_DASHES   = {1: (1, 2), 2: (4, 2, 1, 2), 3: (1, 2)}
DEMOD_ALPHA     = {"complex": 1.0, "simple": 0.55}
DEMOD_LW        = {"complex": 2.0, "simple": 1.4}

os.makedirs(OUT_DIR, exist_ok=True)
cmap = plt.get_cmap("tab10")


def load_eval(path, stage):
    fpath = os.path.join(path, "outputs", f"eval_stage_{stage}.pt")
    if not os.path.exists(fpath):
        return None, None
    d = torch.load(fpath, weights_only=True)
    snr = d["SNR"].float().numpy()
    ber = d["BER"].float()
    return snr, ber


def safe_log(arr):
    return np.where(arr > MIN_BER, arr, np.nan)


# ── collect valid model folders ───────────────────────────────────────────────
valid_indices = []
for idx in MODEL_INDICES:
    p = os.path.join(BASE_PATH, f"{ORIGINAL_NAME}_{idx}")
    if os.path.isdir(p):
        valid_indices.append(idx)
    else:
        print(f"[skip] {ORIGINAL_NAME}_{idx}: folder not found")

if not valid_indices:
    sys.exit("No model folders found.")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title("Worst BER vs SNR — all models & demod types", fontsize=14)
ax.set_xlabel("SNR (dB)", fontsize=12)
ax.set_ylabel("BER", fontsize=12)
ax.set_yscale("log")
ax.set_ylim(bottom=MIN_BER, top=1.0)
if SNR_XLIM is not None:
    ax.set_xlim(SNR_XLIM)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)

# ── plot loop ─────────────────────────────────────────────────────────────────
for i, idx in enumerate(valid_indices):
    color    = cmap(i % 10)
    main_dir = os.path.join(BASE_PATH, f"{ORIGINAL_NAME}_{idx}")

    simple_dir = None
    if QAM_SUB_ROOT:
        qam_root = os.path.join(main_dir, QAM_SUB_ROOT)
        if os.path.isdir(qam_root):
            subs = sorted([
                os.path.join(qam_root, d) for d in os.listdir(qam_root)
                if os.path.isdir(os.path.join(qam_root, d))
            ])
            if subs:
                simple_dir = subs[0]

    variants = [("complex", main_dir)]
    if simple_dir:
        variants.append(("simple", simple_dir))

    for demod, base_dir in variants:
        alpha  = DEMOD_ALPHA[demod]
        lw     = DEMOD_LW[demod]
        dotted = (demod == "simple")

        for stage in STAGES:
            snr, ber = load_eval(base_dir, stage)
            if snr is None:
                print(f"  [skip] M{idx} {demod} stage{stage}: not found in {base_dir}")
                continue

            worst = safe_log(ber.max(dim=0).values.numpy())

            dashes = SIMPLE_DASHES[stage] if dotted else STAGE_DASHES[stage]
            label  = f"M{idx} {demod}"

            ax.plot(snr, worst,
                    linestyle=STAGE_LINESTYLE[stage],
                    dashes=dashes,
                    marker=STAGE_MARKER[stage], markersize=4, markevery=5,
                    color=color, alpha=alpha,
                    linewidth=lw,
                    label=label)

# ── legend ────────────────────────────────────────────────────────────────────
color_handles = [
    mlines.Line2D([], [], color=cmap(i % 10), linewidth=3, label=f"Model {idx}")
    for i, idx in enumerate(valid_indices)
]

stage_handles = [
    mlines.Line2D([], [], color="k",
                  linestyle=STAGE_LINESTYLE[s], marker=STAGE_MARKER[s],
                  markersize=5, linewidth=2, label=f"Stage {s}")
    for s in STAGES
]

demod_handles = [
    mlines.Line2D([], [], color="k", linewidth=2.0, linestyle="-",
                  alpha=1.0,  label="complex demod"),
    mlines.Line2D([], [], color="k", linewidth=1.4, linestyle=":",
                  alpha=0.55, label="simple demod"),
]

leg1 = ax.legend(handles=color_handles,
                 loc="upper right", bbox_to_anchor=(1.0, 1.0),
                 fontsize=8, title="Models", title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)

ax.legend(handles=stage_handles + demod_handles,
          loc="upper right", bbox_to_anchor=(1.0, 0.52),
          fontsize=8, title="Style key", title_fontsize=9, framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, OUT_FILE)
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved → {out_path}")
plt.close(fig)