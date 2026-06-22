"""
plot_relay_blobs.py
===================
Loads all model folders matching  <prefix>_<idx>  and plots, on one figure,
the spatial distribution of relay channel assignments for every model.

For each model × channel, the relay positions assigned to that channel are
summarised as a filled convex-hull "blob" so models with different relay
placements can be compared at a glance.

Usage
-----
    python plot_relay_blobs.py
        --prefix      my_model          # folder prefix  (my_model_0, my_model_1, …)
        --stage       stage_2           # which saved stage to load
        --save        blobs.png         # output path  (optional)
        --models-dir  /path/to/models   # directory containing the model folders
        --no-show                       # suppress interactive window

All arguments are optional; defaults match the script body below.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import torch

import sys
from pathlib import Path
# Add the main folder (parent of code_compare_results, etc.) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from code_Networks.Network_multy_channels import Network_multy_channel, load_model as _load_model

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded configuration  (edit these when running on the server)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PREFIX     = "compare_networks"       # folder prefix  →  model_0, model_1, …
DEFAULT_STAGE      = 2             # integer stage to load (2 = stage_2)
DEFAULT_SAVE_PATH  = "relay_blobs.png"
DEFAULT_MODELS_DIR = "/home/dsi/mishans1/projects/code_for_server_based/simulation/compare_networks"          # None → use the script's own directory
SYSTEM_BW = 0.05          # KDE bandwidth for blob smoothing (0.05–0.3; smaller = tighter to points)
# ─────────────────────────────────────────────────────────────────────────────
# CPU/CUDA compatibility
# ─────────────────────────────────────────────────────────────────────────────
# Models were trained/saved on CUDA nodes. When this script runs on a
# CPU-only machine, torch.load tries to restore tensors to their original
# (CUDA) device and crashes unless map_location is forced to CPU.
# We force this globally, since the project's own load_model()/load() calls
# (in Network_multy_channels.py / Network_single_channel.py) do not always
# pass map_location through.
_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_orig_torch_load = torch.load


def _cpu_safe_torch_load(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs["map_location"] = torch.device("cpu")
    return _orig_torch_load(*args, **kwargs)


torch.load = _cpu_safe_torch_load


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pt(folder, name):
    return torch.load(os.path.join(folder, "data", f"{name}.pt"),
                      weights_only=False)


def find_model_folders(prefix, base=None):
    """Return sorted list of existing folders matching <prefix>_<int>.

    base : str or None
        Directory to search in. Defaults to the script's own directory
        if not provided.
    """
    folders = []
    if base is None:
        base = os.path.dirname(os.path.abspath(__file__))  # script's directory
    base = os.path.abspath(base)
    print(f"Searching for folders in: {base}")
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if not os.path.isdir(full):
            continue
        if not entry.startswith(prefix + "_"):
            continue
        suffix = entry[len(prefix) + 1:]
        if suffix.isdigit():
            folders.append((int(suffix), full))
    folders.sort(key=lambda x: x[0])
    return [f for _, f in folders]


def load_model_data(folder, stage):
    """Load positions and channel assignment using the project's load_model."""
    posR = _pt(folder, "posR").numpy()   # [N_relays, 2]
    posT = _pt(folder, "posT").numpy()   # [N_channels, 2]
    posU = _pt(folder, "posU")           # list or tensor

    model = _load_model(folder, stage)
    model.update_v()
    model.culc_p()

    N_channels = model.N_channels
    P_stack    = np.stack(
        [model.P[c].detach().squeeze().numpy() for c in range(N_channels)],
        axis=0)                          # [C, N_relays]
    assignment = P_stack.argmax(axis=0) # [N_relays]

    return dict(
        posR=posR,
        posT=posT,
        posU=posU,
        assignment=assignment,
        N_channels=N_channels,
        name=os.path.basename(folder),
    )


def draw_kde_blob(ax, pts, color,
                  bw=SYSTEM_BW,
                  threshold=0.05,
                  alpha_fill=0.10,
                  grid_res=300j):
    """
    Draw a tight filled boundary around a 2-D point cloud using KDE.
    bw        : bandwidth — smaller = tighter to points (0.05–0.3)
    threshold : contour level as fraction of peak density
                0.05 = captures almost all points
                0.20 = excludes sparse outliers
    """
    if len(pts) < 3:
        return
    kde = gaussian_kde(pts.T, bw_method=bw)
    margin = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.15
    x0, x1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    y0, y1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    xx, yy = np.mgrid[x0:x1:grid_res, y0:y1:grid_res]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    level = density.max() * threshold
    ax.contourf(xx, yy, density, levels=[level, density.max()],
                colors=[color], alpha=alpha_fill, zorder=1)
    ax.contour(xx, yy, density, levels=[level],
               colors=[color], linewidths=1.5,  linestyle='--', dashes=(5, 1), alpha=0.8, zorder=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def _compute_kde_grid(pts, bw=SYSTEM_BW, grid_res=300j,
                      circle_cx=0.0, circle_cy=0.0, circle_r=None):
    """
    Compute a KDE density grid and mask everything outside the network circle
    to NaN so it plots as white.

    Density is computed as follows:
      1. Fit a Gaussian KDE to the relay positions (x, y).
      2. Evaluate it on a regular 2-D grid covering the network area.
      3. Each grid cell value is proportional to the probability that a
         randomly chosen relay (from the pooled set across all models) falls
         near that cell — i.e. how commonly a relay occupies that area.
      4. Cells outside the network circle are set to NaN so the colourmap
         renders them as white (no colour).
    The bandwidth `bw` controls the smoothing radius: smaller = tighter to
    individual relay dots, larger = smoother regional trends.
    """
    kde    = gaussian_kde(pts.T, bw_method=bw)
    # grid bounds: always the full circle extent so subplots are comparable
    if circle_r is not None:
        pad  = circle_r * 0.12
        x0, x1 = circle_cx - circle_r - pad, circle_cx + circle_r + pad
        y0, y1 = circle_cy - circle_r - pad, circle_cy + circle_r + pad
    else:
        margin = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.15
        x0, x1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
        y0, y1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin

    xx, yy  = np.mgrid[x0:x1:grid_res, y0:y1:grid_res]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # mask outside the circle → NaN → renders as white
    if circle_r is not None:
        outside = (xx - circle_cx)**2 + (yy - circle_cy)**2 > circle_r**2
        density = density.astype(float)
        density[outside] = np.nan

    return xx, yy, density, (x0, x1, y0, y1)


def _draw_circle(ax, cx, cy, r, **kwargs):
    """Draw the network boundary circle."""
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Vibrant colour palette
# ─────────────────────────────────────────────────────────────────────────────
# Just blue and orange — vivid, high-contrast, cycles if N_channels > 2.
VIVID_HEX = [
    "#2979FF",  # vivid blue
    "#FF6D00",  # vivid orange
]

import matplotlib.colors as mcolors
VIVID_RGB = [np.array(mcolors.to_rgb(h)) for h in VIVID_HEX]


def vivid_color(c):
    """Get the vivid hex colour for channel index c (cycles if c exceeds palette)."""
    return VIVID_HEX[c % len(VIVID_HEX)]


def _add_tx_rx(ax, all_data, N_channels, ch_cmap, dot_color=None):
    """Scatter transmitters and users on ax."""
    for data in all_data:
        posT = data["posT"]
        posU = data["posU"]
        for c in range(N_channels):
            if dot_color is not None:
                col = dot_color
            elif isinstance(ch_cmap, list):
                col = ch_cmap[c % len(ch_cmap)]
            else:
                col = ch_cmap(c)
            ax.scatter(1.1*posT[c, 0], 1.1*posT[c, 1],
                       marker="^", color=col, s=1000,
                       zorder=5, edgecolors="k", linewidths=0.7)
            pu = posU[c] if isinstance(posU, list) else posU[c]
            if torch.is_tensor(pu):
                pu = pu.numpy()
            ax.scatter(1.1*pu[:, 0], 1.1*pu[:, 1],
                       marker="s", color=col, s=1000,
                       zorder=5, edgecolors="k", linewidths=0.7)



def plot_blobs(prefix, stage=2, save_path=None, models_dir=None):
    folders = find_model_folders(prefix, base=models_dir)
    if not folders:
        print(f"No folders found matching prefix '{prefix}_<idx>'"
              f"{' in ' + models_dir if models_dir else ''}")
        return

    print(f"Found {len(folders)} model folders: "
          + ", ".join(os.path.basename(f) for f in folders))

    all_data = []
    for folder in folders:
        print(f"  Loading {os.path.basename(folder)} …", end=" ", flush=True)
        try:
            d = load_model_data(folder, stage)
            all_data.append(d)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")

    if not all_data:
        print("No models loaded successfully.")
        return

    N_channels = all_data[0]["N_channels"]
    n_models   = len(all_data)
    ch_cmap    = lambda c: vivid_color(c)   # vivid hex colour per channel

    # ── network circle: relays are drawn uniformly inside radius R centred at
    #    origin; transmitters sit at R_user ≈ R * 1.1.  Estimate R from posR.
    all_posR = np.vstack([d["posR"] for d in all_data])
    circle_r  = float(np.linalg.norm(all_posR, axis=1).max()) * 1.05
    circle_cx, circle_cy = 0.0, 0.0

    # ── accumulate relay positions per channel across all models ──────────────
    ch_pts_all = {c: [] for c in range(N_channels)}
    for data in all_data:
        posR       = data["posR"]
        assignment = data["assignment"]
        for c in range(N_channels):
            pts = posR[assignment == c]
            if len(pts):
                ch_pts_all[c].append(pts)

    # ── compute KDE grids once (shared circle bounds → comparable colour scale)
    kde_grids  = {}
    for c in range(N_channels):
        if not ch_pts_all[c]:
            continue
        all_pts = np.concatenate(ch_pts_all[c], axis=0)
        grid    = _compute_kde_grid(all_pts, bw=SYSTEM_BW, grid_res=300j,
                                    circle_cx=circle_cx,
                                    circle_cy=circle_cy,
                                    circle_r=circle_r)
        kde_grids[c] = (grid, all_pts)

    global_vmax = max(np.nanmax(g[2]) for (g, _) in kde_grids.values())

    # ── output directory: <models_dir or script dir>/<prefix>_results ────────
    results_base = models_dir if models_dir else os.path.dirname(os.path.abspath(__file__))
    results_dir  = os.path.join(results_base, prefix + "_results")
    os.makedirs(results_dir, exist_ok=True)

    base_path = save_path if save_path else "relay_blobs.png"
    base, ext = os.path.splitext(base_path)

 

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — all channels on one heatmap (RGBA composite, white background)
    # Each channel: fixed RGB colour, density drives alpha → stays bright
    # ══════════════════════════════════════════════════════════════════════════
    ch_colors_rgb = VIVID_RGB

    fig2, ax2 = plt.subplots(figsize=(10, 9))
    ax2.set_aspect("equal")
    # ax2.set_title(
    #     f"Relay density — all channels overlaid  ({n_models} models)\n"
    #     f"colour = channel,  intensity = how often a relay is here",
    #     fontsize=12)
    ax2.set_facecolor("white")
    ax2.set_axis_off()
    ax2.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)

    first_grid = next(iter(kde_grids.values()))[0]
    xx0, yy0 = first_grid[0], first_grid[1]
    extent = [xx0.min(), xx0.max(), yy0.min(), yy0.max()]
    H, W   = xx0.shape
    canvas = np.ones((W, H, 4), dtype=float)

    for c, (grid_data, _) in kde_grids.items():
        xx, yy, density, _ = grid_data
        rgb   = ch_colors_rgb[c % len(ch_colors_rgb)]
        d_norm = np.nan_to_num(density, nan=0.0)
        d_norm = (d_norm / global_vmax).clip(0, 1)
        # gamma < 1 boosts mid/low density regions so colour reads vivid
        # sooner instead of fading out to pale near the edges of each blob
        d_boost = d_norm ** 0.55
        alpha_c = d_boost * 0.90
        for ch_idx in range(3):
            canvas[:, :, ch_idx] = (
                canvas[:, :, ch_idx] * (1 - alpha_c.T)
                + rgb[ch_idx] * alpha_c.T
            )

    outside = (xx0 - circle_cx)**2 + (yy0 - circle_cy)**2 > circle_r**2
    canvas[outside.T, :3] = 1.0
    canvas[outside.T,  3] = 1.0

    ax2.imshow(canvas, origin="lower", extent=extent,
               aspect="equal", zorder=1, interpolation="bilinear")
    _draw_circle(ax2, circle_cx, circle_cy, circle_r,
                 color="k", linewidth=1.5, linestyle=(0,(10,10)), alpha=0.6, zorder=4)
    _add_tx_rx(ax2, all_data, N_channels, ch_colors_rgb)

    fig2.tight_layout()
    p2 = f"{base}_density_combined{ext}"
    fig2.savefig(os.path.join(results_dir, p2), dpi=150, bbox_inches="tight")
    print(f"Figure 2 (combined density) saved → {os.path.join(results_dir, p2)}")
    plt.close(fig2)

    ch_seq_cmaps = [
        LinearSegmentedColormap.from_list(f"vivid_{c}", ["#FFFFFF", VIVID_HEX[c % len(VIVID_HEX)]])
        for c in range(N_channels)
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — one heatmap per channel (split subplots, shared colour scale)
    # ══════════════════════════════════════════════════════════════════════════
    ncols = min(N_channels, 3)
    nrows = int(np.ceil(N_channels / ncols))
    fig3, axes = plt.subplots(nrows, ncols,
                               figsize=(6 * ncols, 5.5 * nrows),
                               squeeze=False)
    fig3.suptitle(
        f"Relay assignment density per channel — {n_models} models\n"
        f"(brightness = how often a relay is assigned to this area)",
        fontsize=13, y=1.01)

    for c in range(N_channels):
        row, col = divmod(c, ncols)
        ax = axes[row][col]
        ax.set_aspect("equal")
        ax.set_title(f"Channel {c}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
        ax.set_facecolor("white")

        _draw_circle(ax, circle_cx, circle_cy, circle_r,
                     color="k", linewidth=1.2, linestyle="--", alpha=0.5, zorder=3)

        if c not in kde_grids:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue

        (xx, yy, density, _), all_pts = kde_grids[c]
        seq_cmap = ch_seq_cmaps[c % len(ch_seq_cmaps)]
        seq_cmap.set_bad(color="white", alpha=0)   # NaN outside circle = white

        im = ax.pcolormesh(xx, yy, density,
                           cmap=seq_cmap, shading="auto",
                           vmin=0, vmax=global_vmax, zorder=1)

        level = np.nanmax(density) * 0.05
        ax.contour(xx, yy, density, levels=[level],
                   colors=["white"], linewidths=1.2,
                   linestyles="--", alpha=0.8, zorder=2)

        ax.scatter(all_pts[:, 0], all_pts[:, 1],
                   color="white", alpha=0.25, s=12, zorder=3, linewidths=0)

        _add_tx_rx(ax, all_data, N_channels, ch_cmap, dot_color="white")

        cb = fig3.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("assignment density", fontsize=8)

    for idx in range(N_channels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig3.tight_layout()
    p3 = f"{base}_density_split{ext}"
    fig3.savefig(os.path.join(results_dir, p3), dpi=150, bbox_inches="tight")
    print(f"Figure 3 (split density) saved → {os.path.join(results_dir, p3)}")
    plt.close(fig3)



# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot relay assignment blobs")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"Folder prefix (default: {DEFAULT_PREFIX})")
    parser.add_argument("--stage",  type=int, default=DEFAULT_STAGE,
                        help=f"Stage number to load (default: {DEFAULT_STAGE})")
    parser.add_argument("--save",   default=DEFAULT_SAVE_PATH,
                        help=f"Output file path (default: {DEFAULT_SAVE_PATH})")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                        help="Directory containing the <prefix>_<idx> model "
                             "folders (default: this script's directory)")
    args = parser.parse_args()

    plot_blobs(
        prefix     = args.prefix,
        stage      = args.stage,
        save_path  = args.save,
        models_dir = args.models_dir,
    )