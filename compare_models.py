"""
compare_models.py  —  CLI version
==================================
All analysis runs from the command line.  No GUI required.

Sub-commands
------------
  ber          Evaluate BER vs SNR for one or more model/stage pairs
  log          Plot training log metrics
  variation    Plot BER envelope or all-model curves across variation sub-folders
  arch         Plot network spatial architecture
  const        Plot TX/RX constellation diagrams

Run  python compare_models.py <subcommand> --help  for full options.

Examples
--------
  python compare_models.py ber \\
      --model path/to/model1 --stage 3 --label "Model A" \\
      --model path/to/model2 --stage 1 --label "Model B" \\
      --snr-min 20 --snr-max 40 --snr-step 1 \\
      --save results/ber.png

  python compare_models.py variation \\
      --main path/to/main_model \\
      --subfolders close_sel_dist3 close_sel_dist5 \\
      --snr-min 20 --snr-max 40 \\
      --mode envelope --save results/envelope.png

  python compare_models.py arch \\
      --model path/to/model --stage 3

  python compare_models.py const \\
      --model path/to/model --stage 3 \\
      --snr 10 --itr 50 --save results/const.png
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from Network_multy_channels import Network_multy_channel  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MARKERS = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']
COLORS  = plt.rcParams['axes.prop_cycle'].by_key()['color']

LOG_SCALAR_KEYS = ['loss', 'v_score', 'n_drops', 'worst_BER']
LOG_PER_CH_KEYS = ['BER_per_ch', 'n_relays_per_ch',
                   'w_norm_per_ch', 'b_norm_per_ch', 'p_entropy_per_ch']

METRIC_LABELS = {
    'loss':            'Total loss',
    'v_score':         'V score',
    'n_drops':         'Weight drops / 100 itr',
    'worst_BER':       'Worst BER (all channels)',
    'BER_per_ch':      'Worst BER',
    'n_relays_per_ch': 'Active relays (P>0.9)',
    'w_norm_per_ch':   'Mean |w|',
    'b_norm_per_ch':   'Mean |b|',
    'p_entropy_per_ch':'P entropy (nats)',
}

USE_LOG_SCALE = {'worst_BER', 'BER_per_ch'}

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def dB2lin(x):
    if not torch.is_tensor(x):
        x = torch.tensor(float(x))
    return torch.pow(10.0, x / 10.0)


def read_max_snr_stage1(model_folder):
    path = os.path.join(model_folder, 'data', 'max_snr_train_stage_1.pt')
    if not os.path.isfile(path):
        return None
    try:
        return float(torch.load(path, weights_only=True))
    except Exception:
        return None


def load_model(path, stage):
    def _pt(name):
        return torch.load(os.path.join(path, 'data', name + '.pt'), weights_only=True)
    model = Network_multy_channel(
        N_users           = _pt('N_users'),
        N_relays          = int(_pt('N_relays')),
        N_channels        = int(_pt('N_channels')),
        connectaionMatrix = _pt('connectaionMatrix'),
        MatcgSR           = _pt('MatcgSR'),
        MatcgRR           = _pt('MatcgRR'),
        MatcgRU           = _pt('cgRU'),
        MatcgTU           = _pt('cgTU'),
        modCode_order     = [2 ** int(n) for n in _pt('N_users')],
        demod_type        = _pt('demod_type'),
        N_rx              = _pt('N_rx'),
        N_tx              = _pt('N_tx'),
    )
    model.load(os.path.join(path, ''), f'stage_{stage}')
    model.eval()
    return model


def savefig(fig, path, dpi=150):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
        print(f'Saved: {path}')
    else:
        plt.show()
    plt.close(fig)


def int_to_binary(integers, n_bits):
    mask = 2 ** torch.arange(n_bits - 1, -1, -1)
    return ((integers.unsqueeze(-1) & mask) > 0).int()


# ─────────────────────────────────────────────────────────────────────────────
# BER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, snr_range, batch, num_itr):
    C   = model.N_channels
    out = torch.zeros(C, len(snr_range))
    with torch.no_grad():
        for c in range(C):
            for si, snr in enumerate(snr_range):
                snr_lin = dB2lin(snr)
                model.sub_networks[c].SNR = snr_lin
                acc = 0.0
                for _ in range(num_itr):
                    signal, bits = model.sub_networks[c].modulator(batch)
                    rm   = model.sub_networks[c](signal, bits)
                    pred = model.sub_networks[c].demodulator(rm)
                    worst, _, _ = model.sub_networks[c].BER(bits=bits, pred=pred)
                    acc += float(worst)
                out[c, si] = acc / num_itr
                print(f'  ch{c} SNR={float(snr):.1f} dB  BER={out[c,si]:.3e}')
                if out[c, si] == 0:
                    break
    return out


def plot_ber_comparison(results, snr_range, save_path=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title('BER vs SNR — model comparison', fontsize=14)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER (log scale)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    snr_np = snr_range.numpy()
    for i, entry in enumerate(results):
        name  = entry['name']
        stage = entry['stage']
        ber   = entry['ber']
        C     = ber.shape[0]
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        if C == 1:
            ax.semilogy(snr_np, ber[0].numpy(), marker=marker, color=color,
                        linewidth=1.8, markersize=5,
                        label=f'{name}  (stage {stage})')
        else:
            worst = ber.max(dim=0).values.numpy()
            ax.semilogy(snr_np, worst, marker=marker, color=color,
                        linewidth=2, markersize=5,
                        label=f'{name}  (stage {stage}, worst ch)')
            for c in range(C):
                ax.semilogy(snr_np, ber[c].numpy(), linestyle='--',
                            color=color, linewidth=0.9, alpha=0.55,
                            label=f'{name}  (stage {stage}, ch{c})')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    savefig(fig, save_path)


def cmd_ber(args):
    if not args.model:
        print('ERROR: at least one --model path required'); return

    snr_min = args.snr_min
    if snr_min is None:
        snr_min = read_max_snr_stage1(args.model[0])
        if snr_min is not None:
            snr_min -= 10
            print(f'Auto SNR min: {snr_min:.1f} dB')
        else:
            snr_min = -20.0

    snr_range = torch.arange(snr_min, args.snr_max + args.snr_step * 0.5, args.snr_step)
    results   = []

    for i, path in enumerate(args.model):
        stages = args.stage if args.stage else [3]
        label  = args.label[i] if args.label and i < len(args.label) \
                 else os.path.basename(os.path.normpath(path))
        for stage in stages:
            print(f'\nLoading: {label}  |  stage {stage}')
            try:
                model = load_model(path, stage)
            except Exception as e:
                print(f'ERROR loading model: {e}'); continue
            ber = evaluate_model(model, snr_range, args.batch, args.itr)
            results.append(dict(name=label, stage=stage, ber=ber))
            if args.save:
                base = os.path.splitext(args.save)[0]
                safe = label.replace(' ', '_')
                # torch.save(ber, f'{base}_{safe}_stage{stage}_BER.pt')

    if results:
        plot_ber_comparison(results, snr_range, save_path=args.save)


# ─────────────────────────────────────────────────────────────────────────────
# Log
# ─────────────────────────────────────────────────────────────────────────────

def load_log(filepath):
    data = torch.load(filepath, weights_only=False)
    if 'log' in data and isinstance(data['log'], dict):
        data = data['log']
    return data


def get_series(log, key, channel):
    itr  = np.array(log['itr_axis'])
    vals = np.array(log[key] if channel is None else log[key][channel], dtype=float)
    return itr, vals


def plot_log_comparison(entries, metric_key, metric_channel,
                        window=10, save_path=None):
    use_log = metric_key in USE_LOG_SCALE
    ma_kern = np.ones(window) / window
    fig, ax = plt.subplots(figsize=(12, 5))
    title = METRIC_LABELS.get(metric_key, metric_key)
    if metric_channel is not None:
        title += f' — ch {metric_channel}'
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plot_fn = ax.semilogy if use_log else ax.plot

    for i, entry in enumerate(entries):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        itr, vals = get_series(entry['log'], metric_key, metric_channel)
        if use_log:
            vals = np.where(vals == 0, np.nan, vals)
        plot_fn(itr, vals, color=color, alpha=0.3, linewidth=1)
        valid = ~np.isnan(vals)
        if valid.sum() >= window:
            ma = np.convolve(vals[valid], ma_kern, mode='valid')
            plot_fn(itr[valid][window - 1:], ma, color=color, linewidth=2,
                    marker=marker, markevery=max(1, len(ma) // 15),
                    markersize=5, label=entry['label'])
        else:
            plot_fn(itr[valid], vals[valid], color=color,
                    linewidth=2, marker=marker, markersize=5, label=entry['label'])

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    savefig(fig, save_path)


def cmd_log(args):
    # resolve log files
    log_files  = []
    log_labels = []

    for i, src in enumerate(args.log):
        lbl = args.label[i] if args.label and i < len(args.label) \
              else os.path.basename(os.path.dirname(src)).replace('_', ' ')
        if os.path.isdir(src):
            candidates = [
                os.path.join(src, 'data', 'stage2_all_epochs_raw.pt'),
                os.path.join(src, 'stage2_all_epochs_raw.pt'),
            ]
            found = next((p for p in candidates if os.path.exists(p)), None)
            if found is None:
                print(f'ERROR: cannot find log in {src}'); continue
            log_files.append(found)
        else:
            log_files.append(src)
        log_labels.append(lbl)

    if not log_files:
        print('ERROR: no valid log files found'); return

    entries = []
    for fp, lbl in zip(log_files, log_labels):
        try:
            entries.append({'label': lbl, 'log': load_log(fp)})
            print(f'Loaded log: {lbl}  ({fp})')
        except Exception as e:
            print(f'ERROR loading {fp}: {e}')

    if not entries:
        return

    # list available metrics if requested
    if args.list_metrics:
        log = entries[0]['log']
        print('\nAvailable metrics:')
        for k in LOG_SCALAR_KEYS:
            if k in log:
                print(f'  {k}')
        for k in LOG_PER_CH_KEYS:
            if k in log:
                for ch in sorted(log[k].keys()):
                    print(f'  {k}:{ch}')
        return

    # parse metric
    key = args.metric
    ch  = None
    if ':' in key:
        key, ch = key.split(':', 1)
        ch = int(ch)

    plot_log_comparison(entries, key, ch, window=args.window, save_path=args.save)


# ─────────────────────────────────────────────────────────────────────────────
# Variation
# ─────────────────────────────────────────────────────────────────────────────

def discover_subfolders(main_folder):
    result = []
    if not os.path.isdir(main_folder):
        return result
    for name in sorted(os.listdir(main_folder)):
        path = os.path.join(main_folder, name)
        if not os.path.isdir(path):
            continue
        for model_name in os.listdir(path):
            out = os.path.join(path, model_name, 'outputs')
            if (os.path.isfile(os.path.join(out, 'SNR.pt')) and
                    os.path.isfile(os.path.join(out, 'worst_BER.pt'))):
                result.append(name)
                break
    return result


def discover_models_in_subfolders(main_folder, subfolders):
    found = []
    for sf in subfolders:
        sfp = os.path.join(main_folder, sf)
        if not os.path.isdir(sfp):
            continue
        for model_name in sorted(os.listdir(sfp)):
            vmp = os.path.join(sfp, model_name)
            if not os.path.isdir(vmp):
                continue
            out = os.path.join(vmp, 'outputs')
            if (os.path.isfile(os.path.join(out, 'SNR.pt')) and
                    os.path.isfile(os.path.join(out, 'worst_BER.pt'))):
                found.append({'label': f'{sf} / {model_name}', 'outputs_path': out})
    return found


def load_variation_ber(outputs_path, snr_targets):
    snr_raw = torch.load(os.path.join(outputs_path, 'SNR.pt'), weights_only=True)
    ber_raw = torch.load(os.path.join(outputs_path, 'worst_BER.pt'), weights_only=True)
    snr_np  = (snr_raw.numpy() if torch.is_tensor(snr_raw) else np.array(snr_raw)).flatten()
    ber_np  = (ber_raw.numpy() if torch.is_tensor(ber_raw) else np.array(ber_raw))
    if ber_np.ndim == 2:
        ber_np = ber_np.max(axis=0)
    ber_np = ber_np.flatten()
    matched_snr, matched_ber = [], []
    for t in snr_targets:
        idx = int(np.argmin(np.abs(snr_np - t)))
        matched_snr.append(snr_np[idx])
        matched_ber.append(ber_np[idx])
    return np.array(matched_snr), np.array(matched_ber)


def load_main_ber(folder, snr_targets, main_label=None):
    """Load stage-3 and stage-1 BER from <folder>/outputs/."""
    out        = os.path.join(folder, 'outputs')
    snr_f      = os.path.join(out, 'SNR.pt')
    base_label = main_label or os.path.basename(os.path.normpath(folder))
    if not os.path.isfile(snr_f):
        print(f'  Main model: SNR.pt not found in {out}, skipping.')
        return []
    snr_raw = torch.load(snr_f, weights_only=True)
    snr_np  = (snr_raw.numpy() if torch.is_tensor(snr_raw) else np.array(snr_raw)).flatten()

    results = []
    for stage in (3, 1):
        ber_f = os.path.join(out, f'worst_BER_stage_{stage}.pt')
        if not os.path.isfile(ber_f):
            continue
        ber_raw = torch.load(ber_f, weights_only=True)
        ber_np  = (ber_raw.numpy() if torch.is_tensor(ber_raw) else np.array(ber_raw))
        if ber_np.ndim == 2:
            ber_np = ber_np.max(axis=0)
        ber_np = ber_np.flatten()
        matched_snr, matched_ber = [], []
        for t in snr_targets:
            idx = int(np.argmin(np.abs(snr_np - t)))
            matched_snr.append(snr_np[idx])
            matched_ber.append(ber_np[idx])
        results.append({'label': f'{base_label}  (main stage {stage})',
                        'snr': np.array(matched_snr), 'ber': np.array(matched_ber)})
        print(f'  Main stage {stage} loaded.')
    return results


# ── paper-ready layout constants (edit here to tune output) ──────────────────
PAPER_W, PAPER_H = 7.0, 4.5
FONT_AX, FONT_TICK, FONT_LEG = 11, 10, 8
SAVE_DPI = 300
LEG_LOC, LEG_NCOL = 'upper right', 2
# ─────────────────────────────────────────────────────────────────────────────

def _add_main_bers(ax, main_bers):
    """Overlay main model BER lines and return the lowest non-zero BER."""
    ylim = 1e-10
    styles = [dict(color='black', linestyle='-', marker='o'),
              dict(color='dimgray', linestyle='--', marker='s')]
    for j, mb in enumerate(main_bers or []):
        snr, ber = mb['snr'], mb['ber']
        ber_plot = np.where(ber > 0, ber, np.nan)
        if ~np.isnan(ber_plot).all():
            ylim = max(ylim, ber_plot[~np.isnan(ber_plot)].min())
        ax.semilogy(snr, ber_plot, linewidth=2.0, markersize=5,
                    label=mb['label'], zorder=5, **styles[j % len(styles)])
    return ylim


def plot_variation_all(results, main_bers=None, save_path=None):
    fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
    ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
    ax.set_ylabel('Worst BER', fontsize=FONT_AX)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    for i, entry in enumerate(results):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        snr, ber = entry['snr'], entry['ber']
        mask = ber > 0
        y   = ber if mask.any() else np.full_like(ber, 1e-9)
        lbl = entry['label'] if mask.any() else entry['label'] + '  (all zeros)'
        ax.semilogy(snr[mask] if mask.any() else snr,
                    y[mask]   if mask.any() else y,
                    marker=marker, color=color, linewidth=1.5,
                    markersize=4, label=lbl)
    ylim = _add_main_bers(ax, main_bers)
    ax.set_ylim([ylim, 1])
    ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
              handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
              columnspacing=1.0, framealpha=0.85)
    fig.tight_layout()
    savefig(fig, save_path, dpi=SAVE_DPI)


def plot_variation_envelope(folder_results, main_bers=None, save_path=None):
    import warnings
    fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
    ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
    ax.set_ylabel('Worst BER', fontsize=FONT_AX)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)

    for i, entry in enumerate(folder_results):
        color    = COLORS[i % len(COLORS)]
        snr      = entry['snr']
        mat      = entry['ber_mat']
        n_models = mat.shape[0]
        mat_safe = np.where(mat > 0, mat, np.nan)

        all_nan = np.all(np.isnan(mat_safe), axis=0)
        if all_nan.any():
            print(f'  Warning [{entry["label"]}]: BER=0 at SNR = {snr[all_nan]} dB — skipped.')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    message='All-NaN slice encountered')
            worst = np.nanmax(mat_safe, axis=0)
            best  = np.nanmin(mat_safe, axis=0)
        valid = ~(np.isnan(worst) | np.isnan(best))

        if n_models == 1:
            ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
                        marker='o', markersize=4, label=entry['label'])
        else:
            ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
                        marker='^', markersize=4,
                        label=f'{entry["label"]}  worst')
            ax.semilogy(snr[valid], best[valid], color=color, linewidth=1.5,
                        linestyle='--', marker='v', markersize=4,
                        label=f'{entry["label"]}  best')
            if valid.any():
                ax.fill_between(snr[valid], best[valid], worst[valid],
                                color=color, alpha=0.18)

    ylim = _add_main_bers(ax, main_bers)
    ax.set_ylim([ylim, 1])
    ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
              handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
              columnspacing=1.0, framealpha=0.85)
    fig.tight_layout()
    savefig(fig, save_path, dpi=SAVE_DPI)


def cmd_variation(args):
    folder = args.main
    snr_min = args.snr_min
    if snr_min is None:
        val = read_max_snr_stage1(folder)
        snr_min = (val - 10) if val is not None else -20.0
        print(f'Auto SNR min: {snr_min:.1f} dB')

    snr_targets = np.arange(snr_min, args.snr_max + args.snr_step * 0.5, args.snr_step)

    # resolve sub-folders
    if args.subfolders:
        sf_list = args.subfolders
    else:
        sf_list = discover_subfolders(folder)
        if not sf_list:
            print('No valid sub-folders found.'); return
        print(f'Auto-discovered sub-folders: {sf_list}')

    sf_labels = {}
    if args.sf_labels:
        for pair in args.sf_labels:
            k, v = pair.split('=', 1)
            sf_labels[k] = v

    main_bers = load_main_ber(folder, snr_targets, main_label=args.main_label)

    if args.mode == 'all':
        models = discover_models_in_subfolders(folder, sf_list)
        if not models:
            print('No valid models found.'); return
        results = []
        for m in models:
            sf_part    = m['label'].split(' / ')[0]
            model_part = m['label'].split(' / ', 1)[1] if ' / ' in m['label'] else m['label']
            lbl        = f"{sf_labels.get(sf_part, sf_part)} / {model_part}"
            try:
                snr_m, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
                results.append({'label': lbl, 'snr': snr_m, 'ber': ber_m})
                print(f'  Loaded: {lbl}')
            except Exception as e:
                print(f'  ERROR {lbl}: {e}')
        if results:
            plot_variation_all(results, main_bers=main_bers, save_path=args.save)

    else:  # envelope
        folder_results = []
        for sf_name in sf_list:
            models = discover_models_in_subfolders(folder, [sf_name])
            if not models:
                print(f'  No models in {sf_name}'); continue
            ber_rows = []
            for m in models:
                try:
                    _, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
                    ber_rows.append(ber_m)
                except Exception as e:
                    print(f'  ERROR {m["label"]}: {e}')
            if not ber_rows:
                continue
            snr_rep, _ = load_variation_ber(models[0]['outputs_path'], snr_targets)
            lbl = sf_labels.get(sf_name, sf_name)
            folder_results.append({'label': lbl, 'snr': snr_rep,
                                   'ber_mat': np.vstack(ber_rows)})
            print(f'  {lbl}: {len(ber_rows)} model(s)')
        if folder_results:
            plot_variation_envelope(folder_results, main_bers=main_bers, save_path=args.save)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

def plot_architecture(model, folder_path, save_path=None):
    def _pt(name):
        return torch.load(os.path.join(folder_path, 'data', name + '.pt'),
                          weights_only=False)
    posT = _pt('posT')
    posR = _pt('posR')
    posU = _pt('posU')

    C        = model.N_channels
    N_relays = model.N_relays
    model.update_v()
    model.culc_p()

    color_RR = np.zeros((N_relays, 1))
    for c in range(C):
        pl = model.P[c]
        color_RR[pl.detach().numpy() >= 0.5] = c + 1

    colors = plt.get_cmap('jet', C + 1)
    fig    = plt.figure(figsize=(12, 10))

    for c in range(C):
        col  = colors(c)
        plt.scatter(float(posT[c, 0]), float(posT[c, 1]),
                    marker='^', label=f'TX - channel {c}',
                    color=col, s=300, zorder=5, edgecolors='black', linewidths=0.6)
        mask = np.squeeze(color_RR == c + 1)
        if mask.any():
            plt.scatter(posR[mask, 0].numpy(), posR[mask, 1].numpy(),
                        marker='o', color=col,
                        label=f'relay - channel - {c}', s=200, zorder=3,
                        edgecolors='black', linewidths=0.4)
        pu     = posU[c]
        pu_np  = pu.numpy() if torch.is_tensor(pu) else np.array(pu)
        if pu_np.ndim == 1:
            pu_np = pu_np.reshape(1, -1)
        plt.scatter(pu_np[:, 0], pu_np[:, 1],
                    marker='v', color=col,
                    label=f'RX - channel - {c}', s=300, zorder=5,
                    edgecolors='black', linewidths=0.6)

    plt.xticks([]); plt.yticks([])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncols=3 * C, mode='expand', borderaxespad=0.)
    plt.tight_layout()
    savefig(fig, save_path)


def cmd_arch(args):
    print(f'Loading model from {args.model}, stage {args.stage}')
    model = load_model(args.model, args.stage)
    plot_architecture(model, folder_path=args.model, save_path=args.save)


# ─────────────────────────────────────────────────────────────────────────────
# Constellation
# ─────────────────────────────────────────────────────────────────────────────

def plot_constellation(model, snr_db=10.0, n_itr=1, save_path=None):
    snr_lin  = dB2lin(snr_db)
    C        = model.N_channels
    sym_cmap = plt.get_cmap('tab20')

    for c in range(C):
        sub       = model.sub_networks[c]
        sub.SNR   = snr_lin
        N_users_c = int(model.N_users[c])
        N_symbols = 2 ** N_users_c

        bits = int_to_binary(torch.arange(0, N_symbols), N_users_c).to(torch.complex64)
        if model.demod_type == 'complex':
            s = sub.transmitNN(bits).detach()
        else:
            symbols_int = torch.sum(
                2 ** torch.unsqueeze(
                    torch.linspace(0, N_users_c - 1, N_users_c), dim=1)
                * bits.T, dim=0
            ).real.to(torch.int32)
            s = sub.modulation[symbols_int]

        print(f'Channel {c}: max TX amplitude^2 = {torch.max(s.abs() ** 2).item():.4f}')

        s_plot = s.unsqueeze(1) if model.demod_type == 'simple' else s
        if s_plot.ndim == 1:
            s_plot = s_plot.unsqueeze(1)
        N_tx_plot = s_plot.shape[1]

        rm_runs = []
        with torch.no_grad():
            for _ in range(n_itr):
                if model.demod_type == 'complex':
                    rm = sub(s, bits.real.T).detach().T
                else:
                    rm = torch.squeeze(sub(s, bits.real.T).detach(), 1).T
                rm_runs.append(rm)

        bits_labels = [
            ''.join(str(b) for b in
                    int_to_binary(torch.tensor([sym]), N_users_c)[0].int().tolist())
            for sym in range(N_symbols)
        ]
        sym_colors = [sym_cmap(sym / max(N_symbols - 1, 1)) for sym in range(N_symbols)]

        def _save_ch(fig, suffix):
            if save_path:
                base, ext = os.path.splitext(save_path)
                ext = ext or '.png'
                savefig(fig, f'{base}_ch{c}_{suffix}{ext}')
            else:
                savefig(fig, None)

        # ── TX figure ────────────────────────────────────────────────────────
        fig_tx, axes_tx = plt.subplots(1, N_tx_plot, figsize=(6 * N_tx_plot, 6),
                                       squeeze=False)
        fig_tx.suptitle(f'Channel {c}  |  TX  |  SNR = {snr_db:.1f} dB', fontsize=13)
        for tx_idx in range(N_tx_plot):
            ax = axes_tx[0, tx_idx]
            for sym in range(N_symbols):
                ax.scatter(s_plot[sym, tx_idx].real.item(),
                           s_plot[sym, tx_idx].imag.item(),
                           color=sym_colors[sym], s=100, zorder=3)
                ax.annotate(bits_labels[sym],
                            (s_plot[sym, tx_idx].real.item(),
                             s_plot[sym, tx_idx].imag.item()),
                            textcoords='offset points', xytext=(6, 6),
                            fontsize=9, fontweight='bold')
            ax.set_title(f'TX antenna {tx_idx}', fontsize=11)
            ax.set_xlabel('Real'); ax.set_ylabel('Imag')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
                       for s in range(N_symbols)]
            ax.legend(handles=handles, title='Symbol', fontsize=7,
                      ncol=max(1, N_symbols // 4), loc='best')
        fig_tx.tight_layout()
        _save_ch(fig_tx, 'TX')

        # ── RX figures (one per user) ─────────────────────────────────────────
        for u in range(N_users_c):
            fig_rx, ax = plt.subplots(figsize=(7, 7))
            fig_rx.suptitle(
                f'Channel {c}  |  RX user {u}  |  SNR = {snr_db:.1f} dB  '
                f'({n_itr} iteration{"s" if n_itr > 1 else ""})', fontsize=13)
            for sym in range(N_symbols):
                pts = torch.stack([rm_runs[it][sym, u] for it in range(n_itr)])
                ax.scatter(pts.real.numpy(), pts.imag.numpy(),
                           color=sym_colors[sym], s=20, alpha=0.6, zorder=3)
                cx = pts.real.mean().item()
                cy = pts.imag.mean().item()
                ax.scatter([cx], [cy], color=sym_colors[sym], s=80,
                           marker='+', zorder=5, linewidths=2)
                ax.annotate(bits_labels[sym], (cx, cy),
                            textcoords='offset points', xytext=(6, 6),
                            fontsize=9, fontweight='bold', color=sym_colors[sym])
            ax.set_xlabel('Real', fontsize=10); ax.set_ylabel('Imag', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
                       for s in range(N_symbols)]
            ax.legend(handles=handles, title='Symbol', fontsize=8,
                      ncol=max(1, N_symbols // 4), loc='best')
            fig_rx.tight_layout()
            _save_ch(fig_rx, f'RX_user{u}')


def cmd_const(args):
    print(f'Loading model from {args.model}, stage {args.stage}')
    model = load_model(args.model, args.stage)
    plot_constellation(model, snr_db=args.snr, n_itr=args.itr, save_path=args.save)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Interactive wizard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt, default=None):
    """
    Ask a question.  If default is given, show it and allow empty-Enter to accept.
    Returns the user's string (stripped), or the default if Enter pressed.
    """
    if default is not None:
        shown = f'  {prompt} [{default}]: '
    else:
        shown = f'  {prompt}: '
    while True:
        try:
            raw = input(shown).strip()
        except (EOFError, KeyboardInterrupt):
            print('\nAborted.')
            sys.exit(0)
        if raw == '' and default is not None:
            return str(default)
        if raw != '':
            return raw
        print('  (value required)')


def _ask_yn(prompt, default=True):
    """Yes/no question.  Returns bool."""
    suffix = '[Y/n]' if default else '[y/N]'
    raw = _ask(f'{prompt} {suffix}', default='y' if default else 'n')
    return raw.lower() in ('y', 'yes', '1')


def _ask_default(prompt, default):
    """Ask whether to keep a default; if not, ask for new value."""
    if _ask_yn(f'{prompt} (default: {default}) — keep?', default=True):
        return default
    return _ask(prompt)


def _ask_path(prompt, must_exist=True):
    """Ask for a filesystem path, optionally verifying it exists."""
    while True:
        val = _ask(prompt)
        if not must_exist:
            return val
        if os.path.exists(val):
            return val
        print(f'  Path not found: {val}')


def _ask_save():
    """Ask for an optional save path."""
    if _ask_yn('Save plot to file?', default=False):
        return _ask('Save path (e.g. results/plot.png)', default=None)
    return None


def _ask_stage_multi():
    """Ask which stages to use (checkboxes style)."""
    print('  Which stage(s) to evaluate?')
    stages = []
    for s in (1, 2, 3):
        if _ask_yn(f'    Stage {s}?', default=(s == 3)):
            stages.append(s)
    if not stages:
        print('  (no stage selected, defaulting to stage 3)')
        stages = [3]
    return stages


def _separator():
    print('\n' + '─' * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tab wizards
# ─────────────────────────────────────────────────────────────────────────────

def wizard_ber():
    _separator()
    print('  BER COMPARISON')
    print('  Evaluate BER vs SNR for one or more model/stage pairs.')
    _separator()

    # collect models
    models, labels = [], []
    print('\n  Add model folders (empty path to stop).')
    idx = 1
    while True:
        path = _ask(f'Model {idx} folder (or Enter to stop)' if idx > 1
                    else 'Model 1 folder', default=None if idx == 1 else '')
        if path == '' and idx > 1:
            break
        if not os.path.exists(path):
            print(f'  Path not found: {path}')
            continue
        lbl = _ask('  Label for this model',
                   default=os.path.basename(os.path.normpath(path)))
        models.append(path)
        labels.append(lbl)
        idx += 1
        if idx > 1:
            if not _ask_yn('\n  Add another model?', default=False):
                break

    stages = _ask_stage_multi()

    # SNR range
    print()
    snr_min_auto = read_max_snr_stage1(models[0]) if models else None
    snr_min_default = int(round(snr_min_auto - 10)) if snr_min_auto else -20
    snr_min  = float(_ask_default('SNR min (dB)', snr_min_default))
    snr_max  = float(_ask_default('SNR max (dB)', 40))
    snr_step = float(_ask_default('SNR step (dB)', 1))
    batch    = int(_ask_default('Batch size', 1000))
    itr      = int(_ask_default('Avg iterations per SNR point', 10))
    save     = _ask_save()

    # build args namespace and run
    class A: pass
    args = A()
    args.model = models
    args.label = labels
    args.stage = stages
    args.snr_min  = snr_min
    args.snr_max  = snr_max
    args.snr_step = snr_step
    args.batch    = batch
    args.itr      = itr
    args.save     = save
    cmd_ber(args)


def wizard_log():
    _separator()
    print('  LOG COMPARISON')
    print('  Plot training metrics from one or more log files.')
    _separator()

    logs, labels = [], []
    print('\n  Add log files or model folders (empty to stop).')
    idx = 1
    while True:
        path = _ask(f'Log {idx} path (file or model folder)' if idx > 1
                    else 'Log 1 path (file or model folder)', default=None if idx == 1 else '')
        if path == '' and idx > 1:
            break
        if not os.path.exists(path):
            print(f'  Path not found: {path}')
            continue
        lbl = _ask('  Label', default=os.path.basename(os.path.normpath(path)).replace('_', ' '))
        logs.append(path)
        labels.append(lbl)
        idx += 1
        if not _ask_yn('\n  Add another log?', default=False):
            break

    # show available metrics from first log
    print('\n  Loading first log to show available metrics…')
    try:
        fp = logs[0]
        if os.path.isdir(fp):
            for cand in [os.path.join(fp, 'data', 'stage2_all_epochs_raw.pt'),
                         os.path.join(fp, 'stage2_all_epochs_raw.pt')]:
                if os.path.exists(cand):
                    fp = cand; break
        log0 = load_log(fp)
        print('\n  Available metrics:')
        metric_list = []
        for k in LOG_SCALAR_KEYS:
            if k in log0:
                metric_list.append(k)
                print(f'    {len(metric_list):2d})  {k}')
        for k in LOG_PER_CH_KEYS:
            if k in log0:
                for ch in sorted(log0[k].keys()):
                    key_str = f'{k}:{ch}'
                    metric_list.append(key_str)
                    print(f'    {len(metric_list):2d})  {key_str}')
        print()
        choice = _ask('Metric number or key', default='worst_BER')
        if choice.isdigit() and 1 <= int(choice) <= len(metric_list):
            metric = metric_list[int(choice) - 1]
        else:
            metric = choice
    except Exception as e:
        print(f'  Could not read metrics: {e}')
        metric = _ask('Metric key', default='worst_BER')

    window = int(_ask_default('Moving-average window', 10))
    save   = _ask_save()

    class A: pass
    args = A()
    args.log    = logs
    args.label  = labels
    args.metric = metric
    args.list_metrics = False
    args.window = window
    args.save   = save
    cmd_log(args)


def wizard_variation():
    _separator()
    print('  VARIATION COMPARISON')
    print('  Compare BER across variation sub-folders of a main model.')
    _separator()

    main_folder = _ask_path('\n  Main model folder')
    main_label  = _ask('  Main model legend label',
                       default=os.path.basename(os.path.normpath(main_folder)))

    # discover sub-folders
    sf_all = discover_subfolders(main_folder)
    selected_sf = []
    sf_labels   = {}

    if sf_all:
        print(f'\n  Found {len(sf_all)} sub-folder(s):')
        for i, sf in enumerate(sf_all, 1):
            print(f'    {i:2d})  {sf}')
        print()
        if _ask_yn('  Include all sub-folders?', default=True):
            selected_sf = sf_all[:]
        else:
            for sf in sf_all:
                if _ask_yn(f'    Include "{sf}"?', default=True):
                    selected_sf.append(sf)

        print()
        if _ask_yn('  Customise legend labels for sub-folders?', default=False):
            for sf in selected_sf:
                lbl = _ask(f'    Label for "{sf}"', default=sf)
                if lbl != sf:
                    sf_labels[sf] = lbl
    else:
        print('  No sub-folders auto-discovered.')
        raw = _ask('  Enter sub-folder names manually (space-separated)')
        selected_sf = raw.split()

    # SNR range
    print()
    snr_min_auto = read_max_snr_stage1(main_folder)
    snr_min_default = int(round(snr_min_auto - 10)) if snr_min_auto else -20
    snr_min  = float(_ask_default('SNR min (dB)', snr_min_default))
    snr_max  = float(_ask_default('SNR max (dB)', 40))
    snr_step = float(_ask_default('SNR step (dB)', 1))

    mode = 'envelope'
    if _ask_yn('  Plot mode — envelope (best/worst band)? [Y=envelope / n=all curves]',
               default=True):
        mode = 'envelope'
    else:
        mode = 'all'

    save = _ask_save()

    class A: pass
    args = A()
    args.main        = main_folder
    args.main_label  = main_label
    args.subfolders  = selected_sf or None
    args.sf_labels   = [f'{k}={v}' for k, v in sf_labels.items()] or None
    args.snr_min     = snr_min
    args.snr_max     = snr_max
    args.snr_step    = snr_step
    args.mode        = mode
    args.save        = save
    cmd_variation(args)


def wizard_arch():
    _separator()
    print('  ARCHITECTURE')
    print('  Plot the spatial relay network diagram.')
    _separator()

    model_path = _ask_path('\n  Model folder')
    stage      = int(_ask_default('Stage', 3))
    save       = _ask_save()

    class A: pass
    args = A()
    args.model = model_path
    args.stage = stage
    args.save  = save
    cmd_arch(args)


def wizard_const():
    _separator()
    print('  CONSTELLATION')
    print('  Plot TX symbol constellation and received symbol clouds.')
    _separator()

    model_path = _ask_path('\n  Model folder')
    stage      = int(_ask_default('Stage', 3))
    snr        = float(_ask_default('SNR (dB)', 10))
    itr        = int(_ask_default('Noise averaging iterations (1 = clean plot)', 1))
    save       = _ask_save()

    class A: pass
    args = A()
    args.model = model_path
    args.stage = stage
    args.snr   = snr
    args.itr   = itr
    args.save  = save
    cmd_const(args)


# ─────────────────────────────────────────────────────────────────────────────
# Main menu
# ─────────────────────────────────────────────────────────────────────────────

TABS = [
    ('BER Comparison',             wizard_ber),
    ('Log Comparison',             wizard_log),
    ('Variation Comparison',       wizard_variation),
    ('Architecture',               wizard_arch),
    ('Constellation',              wizard_const),
]


def main():
    # If called with CLI args (e.g. from a script), fall through to argparse
    if len(sys.argv) > 1:
        _main_argparse()
        return

    # Interactive menu
    while True:
        print('\n' + '═' * 60)
        print('  MODEL COMPARISON TOOL')
        print('═' * 60)
        for i, (name, _) in enumerate(TABS, 1):
            print(f'  {i})  {name}')
        print('  q)  Quit')
        print('─' * 60)

        try:
            choice = input('  Select tab: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break

        if choice == 'q':
            print('Bye.')
            break

        if choice.isdigit() and 1 <= int(choice) <= len(TABS):
            _, wizard = TABS[int(choice) - 1]
            try:
                wizard()
            except KeyboardInterrupt:
                print('\n  (cancelled — back to menu)')
            except Exception as e:
                print(f'\n  ERROR: {e}')
                import traceback; traceback.print_exc()
            input('\n  Press Enter to return to menu…')
        else:
            print('  Invalid choice.')


def _main_argparse():
    """Legacy argparse interface — use  python compare_models.py <cmd> --help."""
    p = argparse.ArgumentParser(
        prog='compare_models.py',
        description='Run without arguments for interactive mode.')
    sub = p.add_subparsers(dest='cmd', required=True)

    pb = sub.add_parser('ber')
    pb.add_argument('--model',    action='append', required=True)
    pb.add_argument('--stage',    type=int, action='append')
    pb.add_argument('--label',    action='append')
    pb.add_argument('--snr-min',  type=float, default=None)
    pb.add_argument('--snr-max',  type=float, default=40.0)
    pb.add_argument('--snr-step', type=float, default=1.0)
    pb.add_argument('--batch',    type=int,   default=1000)
    pb.add_argument('--itr',      type=int,   default=10)
    pb.add_argument('--save',     default=None)

    pl = sub.add_parser('log')
    pl.add_argument('--log',          action='append', required=True)
    pl.add_argument('--label',        action='append')
    pl.add_argument('--metric',       default='worst_BER')
    pl.add_argument('--list-metrics', action='store_true')
    pl.add_argument('--window',       type=int, default=10)
    pl.add_argument('--save',         default=None)

    pv = sub.add_parser('variation')
    pv.add_argument('--main',        required=True)
    pv.add_argument('--main-label',  default=None)
    pv.add_argument('--subfolders',  nargs='+')
    pv.add_argument('--sf-labels',   nargs='+')
    pv.add_argument('--snr-min',     type=float, default=None)
    pv.add_argument('--snr-max',     type=float, default=40.0)
    pv.add_argument('--snr-step',    type=float, default=1.0)
    pv.add_argument('--mode',        choices=['all', 'envelope'], default='envelope')
    pv.add_argument('--save',        default=None)

    pa = sub.add_parser('arch')
    pa.add_argument('--model', required=True)
    pa.add_argument('--stage', type=int, default=3)
    pa.add_argument('--save',  default=None)

    pc = sub.add_parser('const')
    pc.add_argument('--model', required=True)
    pc.add_argument('--stage', type=int, default=3)
    pc.add_argument('--snr',   type=float, default=10.0)
    pc.add_argument('--itr',   type=int,   default=1)
    pc.add_argument('--save',  default=None)

    args = p.parse_args()
    dispatch = {'ber': cmd_ber, 'log': cmd_log, 'variation': cmd_variation,
                'arch': cmd_arch, 'const': cmd_const}
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()