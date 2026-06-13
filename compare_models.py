# """
# compare_models.py
# =================
# Two-tab GUI tool for offline model analysis.
#
# Tab 1 – BER Comparison
#     Load trained model folders, evaluate BER vs SNR, overlay results.
#
# Tab 2 – Log Comparison
#     Load stage-2 log .pt files (stage2_all_epochs_raw.pt) from one or more
#     models and overlay any logged metric on a single graph.
#     Special case: when only ONE model is loaded, two log files from that
#     model can be selected and plotted together (e.g. two training runs).
# """
#
# import os
# import sys
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
#
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np
# import torch
#
# _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# if _SRC_DIR not in sys.path:
#     sys.path.insert(0, _SRC_DIR)
#
# from Network_multy_channels import Network_multy_channel  # noqa: E402
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────────────────────────────────────
#
# MARKERS = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']
# COLORS  = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# # All keys present in the log dict that can be plotted.
# # (channel-indexed ones are expanded dynamically after loading)
# LOG_SCALAR_KEYS = ['loss', 'v_score', 'n_drops', 'worst_BER']
# LOG_PER_CH_KEYS = ['BER_per_ch', 'n_relays_per_ch',
#                    'w_norm_per_ch', 'b_norm_per_ch', 'p_entropy_per_ch']
#
# METRIC_LABELS = {
#     'loss':           'Total loss',
#     'v_score':        'V score',
#     'n_drops':        'Weight drops / 100 itr',
#     'worst_BER':      'Worst BER (all channels)',
#     'BER_per_ch':     'Worst BER',
#     'n_relays_per_ch':'Active relays (P>0.9)',
#     'w_norm_per_ch':  'Mean |w|',
#     'b_norm_per_ch':  'Mean |b|',
#     'p_entropy_per_ch':'P entropy (nats)',
# }
#
# USE_LOG_SCALE = {'worst_BER', 'BER_per_ch'}
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Shared utility
# # ─────────────────────────────────────────────────────────────────────────────
#
# def read_max_snr_stage1(model_folder: str) -> float | None:
#     """
#     Read max_snr_train_stage_1.pt from <model_folder>/data/.
#     Returns the float value, or None if the file doesn't exist.
#     """
#     path = os.path.join(model_folder, 'data', 'max_snr_train_stage_1.pt')
#     if not os.path.isfile(path):
#         return None
#     try:
#         val = torch.load(path, weights_only=True)
#         return float(val)
#     except Exception:
#         return None
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # BER helpers  (unchanged from original)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def dB2lin(x):
#     if not torch.is_tensor(x):
#         x = torch.tensor(float(x))
#     return torch.pow(10.0, x / 10.0)
#
#
# def load_model(path, stage):
#     def _pt(name):
#         return torch.load(os.path.join(path, 'data', name + '.pt'), weights_only=True)
#     MatcgRR           = _pt('MatcgRR')
#     MatcgSR           = _pt('MatcgSR')
#     cgRU              = _pt('cgRU')
#     connectaionMatrix = _pt('connectaionMatrix')
#     N_users           = _pt('N_users')
#     N_channels        = int(_pt('N_channels'))
#     N_relays          = int(_pt('N_relays'))
#     demod_type        = _pt('demod_type')
#     N_rx             = _pt('N_rx')
#     N_tx            = _pt('N_tx')
#     cgTU             = _pt('cgTU')
#     modCode_order     = [2 ** int(n) for n in N_users]
#     model = Network_multy_channel(N_users=N_users,
#                                   N_relays=N_relays,
#                                   N_channels=N_channels,
#                                   connectaionMatrix=connectaionMatrix,
#                                   MatcgSR=MatcgSR,
#                                   MatcgRR=MatcgRR,
#                                   MatcgRU=cgRU,
#                                   MatcgTU=cgTU,
#                                   modCode_order=modCode_order,
#                                   demod_type=demod_type,
#                                   N_rx=N_rx,
#                                   N_tx=N_tx
#                                   )
#     model.load(path + '\\', f'stage_{stage}')
#     model.eval()
#     return model
#
#
# def evaluate_model(model, snr_range, batch, num_itr):
#     C   = model.N_channels
#     out = torch.zeros(C, len(snr_range))
#     with torch.no_grad():
#         for c in range(C):
#             for si, snr in enumerate(snr_range):
#                 snr_lin = dB2lin(snr)
#                 model.sub_networks[c].SNR = snr_lin
#                 acc = 0.0
#                 for _ in range(num_itr):
#                     signal,bits = model.sub_networks[c].modulator(batch)
#                     rm   = model.sub_networks[c](signal,bits)
#                     pred = model.sub_networks[c].demodulator(rm)
#                     worst, _, _ = model.sub_networks[c].BER(bits=bits, pred=pred)
#                     acc += float(worst)
#                 out[c, si] = acc / num_itr
#                 print(f'  ch{c} SNR={snr:.1f} dB  BER={out[c,si]:.3e}')
#                 if out[c, si] == 0:
#                     break
#     return out
#
#
# def plot_ber_comparison(results, snr_range, save_path=None, show=True):
#     fig, ax = plt.subplots(figsize=(11, 6))
#     ax.set_title('BER vs SNR — model comparison', fontsize=14)
#     ax.set_xlabel('SNR (dB)', fontsize=12)
#     ax.set_ylabel('BER (log scale)', fontsize=12)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
#     snr_np    = snr_range.numpy()
#     color_idx = 0
#     for entry in results:
#         name  = entry['name']
#         stage = entry['stage']
#         ber   = entry['ber']
#         C     = ber.shape[0]
#         color  = COLORS[color_idx % len(COLORS)]
#         marker = MARKERS[color_idx % len(MARKERS)]
#         color_idx += 1
#         if C == 1:
#             ax.semilogy(snr_np, ber[0].numpy(), marker=marker,
#                         color=color, linewidth=1.8, markersize=5,
#                         label=f'{name}  (stage {stage})')
#         else:
#             worst_overall = ber.max(dim=0).values.numpy()
#             ax.semilogy(snr_np, worst_overall, marker=marker,
#                         color=color, linewidth=2, markersize=5,
#                         label=f'{name}  (stage {stage}, worst ch)')
#             for c in range(C):
#                 ax.semilogy(snr_np, ber[c].numpy(), linestyle='--',
#                             color=color, linewidth=0.9, alpha=0.55,
#                             label=f'{name}  (stage {stage}, ch{c})')
#     ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
#               borderaxespad=0, fontsize=9)
#     fig.tight_layout(rect=[0, 0, 0.78, 1])
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f'Plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Log helpers
# # ─────────────────────────────────────────────────────────────────────────────
#
# def load_log(filepath):
#     """Load a stage2_all_epochs_raw.pt log dict."""
#     data = torch.load(filepath, weights_only=False)
#     # unwrap if saved inside a wrapper dict (stage2_epochXX_raw.pt format)
#     if 'log' in data and isinstance(data['log'], dict):
#         data = data['log']
#     return data
#
#
# def expand_metric_names(log):
#     """
#     Return a list of (display_name, key, channel_or_None) tuples
#     for every plottable series in the log.
#     """
#     names = []
#     for k in LOG_SCALAR_KEYS:
#         if k in log:
#             names.append((METRIC_LABELS.get(k, k), k, None))
#     for k in LOG_PER_CH_KEYS:
#         if k in log:
#             ch_dict = log[k]
#             for c in sorted(ch_dict.keys()):
#                 display = f'{METRIC_LABELS.get(k, k)} — ch {c}'
#                 names.append((display, k, c))
#     return names
#
#
# def get_series(log, key, channel):
#     """Extract (itr_array, values_array) from a log dict."""
#     itr = np.array(log['itr_axis'])
#     if channel is None:
#         vals = np.array(log[key], dtype=float)
#     else:
#         vals = np.array(log[key][channel], dtype=float)
#     return itr, vals
#
#
# def plot_log_comparison(entries, metric_key, metric_channel,
#                         window=10, save_path=None, show=True):
#     """
#     entries : list of {label, log}
#     metric_key / metric_channel : which series to plot
#     """
#     use_log  = metric_key in USE_LOG_SCALE
#     ma_kern  = np.ones(window) / window
#     fig, ax  = plt.subplots(figsize=(12, 5))
#     title    = METRIC_LABELS.get(metric_key, metric_key)
#     if metric_channel is not None:
#         title += f' — ch {metric_channel}'
#     ax.set_title(title, fontsize=13)
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel(title)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
#
#     plot_fn = ax.semilogy if use_log else ax.plot
#     color_idx = 0
#
#     for entry in entries:
#         label = entry['label']
#         log   = entry['log']
#         color  = COLORS[color_idx % len(COLORS)]
#         marker = MARKERS[color_idx % len(MARKERS)]
#         color_idx += 1
#
#         itr, vals = get_series(log, metric_key, metric_channel)
#         # mask zeros for log-scale plots
#         if use_log:
#             vals = np.where(vals == 0, np.nan, vals)
#
#         plot_fn(itr, vals, color=color, alpha=0.3, linewidth=1)
#         valid = ~np.isnan(vals)
#         if valid.sum() >= window:
#             ma = np.convolve(vals[valid], ma_kern, mode='valid')
#             plot_fn(itr[valid][window - 1:], ma,
#                     color=color, linewidth=2,
#                     marker=marker, markevery=max(1, len(ma)//15),
#                     markersize=5, label=label)
#         else:
#             plot_fn(itr[valid], vals[valid], color=color,
#                     linewidth=2, marker=marker, markersize=5, label=label)
#
#     ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
#               borderaxespad=0, fontsize=9)
#     fig.tight_layout(rect=[0, 0, 0.80, 1])
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f'Log plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# def _plot_dual_metric(entry, m1_key, m1_ch, m2_key, m2_ch,
#                       window=10, save_path=None, show=True):
#     """
#     Plot two metrics from a single log on twin y-axes.
#     Left axis  = metric 1 (blue tones)
#     Right axis = metric 2 (orange tones)
#     """
#     log      = entry['log']
#     label    = entry['label']
#     ma_kern  = np.ones(window) / window
#
#     itr = np.array(log['itr_axis'])
#
#     def _prep(key, ch):
#         _, vals = get_series(log, key, ch)
#         if key in USE_LOG_SCALE:
#             vals = np.where(vals == 0, np.nan, vals)
#         return vals
#
#     v1 = _prep(m1_key, m1_ch)
#     v2 = _prep(m2_key, m2_ch)
#
#     t1 = METRIC_LABELS.get(m1_key, m1_key) + (f' — ch {m1_ch}' if m1_ch is not None else '')
#     t2 = METRIC_LABELS.get(m2_key, m2_key) + (f' — ch {m2_ch}' if m2_ch is not None else '')
#
#     fig, ax1 = plt.subplots(figsize=(12, 5))
#     fig.suptitle(f'{label}', fontsize=12)
#
#     c1 = COLORS[0]   # blue family
#     c2 = COLORS[1]   # orange family
#
#     # ── left axis ─────────────────────────────────────────────────────────────
#     pf1 = ax1.semilogy if m1_key in USE_LOG_SCALE else ax1.plot
#     ax1.set_xlabel('Iteration')
#     ax1.set_ylabel(t1, color=c1)
#     ax1.tick_params(axis='y', labelcolor=c1)
#
#     pf1(itr, v1, color=c1, alpha=0.25, linewidth=1)
#     valid1 = ~np.isnan(v1)
#     if valid1.sum() >= window:
#         ma1 = np.convolve(v1[valid1], ma_kern, mode='valid')
#         pf1(itr[valid1][window - 1:], ma1, color=c1, linewidth=2,
#             marker='o', markevery=max(1, len(ma1) // 15),
#             markersize=5, label=t1)
#     else:
#         pf1(itr[valid1], v1[valid1], color=c1, linewidth=2, label=t1)
#
#     # ── right axis ────────────────────────────────────────────────────────────
#     ax2 = ax1.twinx()
#     pf2 = ax2.semilogy if m2_key in USE_LOG_SCALE else ax2.plot
#     ax2.set_ylabel(t2, color=c2)
#     ax2.tick_params(axis='y', labelcolor=c2)
#
#     pf2(itr, v2, color=c2, alpha=0.25, linewidth=1)
#     valid2 = ~np.isnan(v2)
#     if valid2.sum() >= window:
#         ma2 = np.convolve(v2[valid2], ma_kern, mode='valid')
#         pf2(itr[valid2][window - 1:], ma2, color=c2, linewidth=2,
#             linestyle='--', marker='^',
#             markevery=max(1, len(ma2) // 15),
#             markersize=5, label=t2)
#     else:
#         pf2(itr[valid2], v2[valid2], color=c2, linewidth=2,
#             linestyle='--', label=t2)
#
#     # combined legend outside
#     h1, l1 = ax1.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     ax1.legend(h1 + h2, l1 + l2, loc='upper left',
#                bbox_to_anchor=(1.08, 1), borderaxespad=0, fontsize=9)
#
#     ax1.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
#     fig.tight_layout(rect=[0, 0, 0.80, 1])
#
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f'Dual-metric plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Variation helpers  (Tab 3)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def discover_subfolders(main_folder: str) -> list[str]:
#     """
#     Return sorted list of immediate sub-folder names inside *main_folder*
#     that contain at least one model with outputs/SNR.pt + outputs/worst_BER.pt.
#     """
#     result = []
#     if not os.path.isdir(main_folder):
#         return result
#     for name in sorted(os.listdir(main_folder)):
#         path = os.path.join(main_folder, name)
#         if not os.path.isdir(path):
#             continue
#         for model_name in os.listdir(path):
#             out = os.path.join(path, model_name, 'outputs')
#             if (os.path.isfile(os.path.join(out, 'SNR.pt')) and
#                     os.path.isfile(os.path.join(out, 'worst_BER.pt'))):
#                 result.append(name)
#                 break
#     return result
#
#
# def discover_models_in_subfolders(main_folder: str,
#                                    subfolders: list[str]) -> list[dict]:
#     """
#     For each sub-folder name in *subfolders*, walk its model directories and
#     collect every model that has outputs/SNR.pt + outputs/worst_BER.pt.
#
#     Returns a list of dicts:
#         { 'label':        '<subfolder> / <model_name>',
#           'outputs_path': full path to the outputs folder }
#     """
#     found = []
#     for sf in subfolders:
#         sfp = os.path.join(main_folder, sf)
#         if not os.path.isdir(sfp):
#             continue
#         for model_name in sorted(os.listdir(sfp)):
#             vmp = os.path.join(sfp, model_name)
#             if not os.path.isdir(vmp):
#                 continue
#             out = os.path.join(vmp, 'outputs')
#             if (os.path.isfile(os.path.join(out, 'SNR.pt')) and
#                     os.path.isfile(os.path.join(out, 'worst_BER.pt'))):
#                 found.append({'label': f'{sf} / {model_name}',
#                               'outputs_path': out})
#     return found
#
#
# def load_variation_ber(outputs_path: str,
#                        snr_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Load SNR.pt and worst_BER.pt from *outputs_path*.
#     For each value in *snr_targets* find the closest recorded SNR point and
#     return matched (snr_matched, ber_matched) arrays.
#
#     worst_BER.pt may be shape [N_snr] or [N_channels, N_snr]; the latter is
#     reduced to the worst channel (max over axis 0).
#     """
#     snr_raw = torch.load(os.path.join(outputs_path, 'SNR.pt'), weights_only=True)
#     ber_raw = torch.load(os.path.join(outputs_path, 'worst_BER.pt'), weights_only=True)
#
#     snr_np = (snr_raw.numpy() if torch.is_tensor(snr_raw) else np.array(snr_raw)).flatten()
#     ber_np = (ber_raw.numpy() if torch.is_tensor(ber_raw) else np.array(ber_raw))
#     if ber_np.ndim == 2:
#         ber_np = ber_np.max(axis=0)
#     ber_np = ber_np.flatten()
#
#     matched_snr, matched_ber = [], []
#     for t in snr_targets:
#         idx = int(np.argmin(np.abs(snr_np - t)))
#         matched_snr.append(snr_np[idx])
#         matched_ber.append(ber_np[idx])
#     return np.array(matched_snr), np.array(matched_ber)
#
#
# def plot_variation_all(results: list[dict], main_bers=None, save_path=None, show=True):
#     """
#     Plot every individual model BER curve, plus optionally the main model.
#     results:  list of { 'label', 'snr': np.ndarray, 'ber': np.ndarray }
#     main_bers: list of { 'label', 'snr': np.ndarray, 'ber': np.ndarray } or None
#
#     Paper-ready formatting:
#       • Double-column IEEE width (7 in × 4.5 in)
#       • Legend inside the axes (upper right) — ncol=2 if many entries
#       • Font sizes matching a 10 pt document
#       • 300 dpi / vector-friendly tight layout
#     To adjust: change PAPER_W/PAPER_H, FONT_* constants, or legend loc/ncol below.
#     """
#     # ── paper layout constants ── tweak these to taste ──────────────────────
#     PAPER_W   = 7.0          # figure width  in inches (3.5 = single col, 7 = double col)
#     PAPER_H   = 4.5          # figure height in inches
#     FONT_AX   = 11           # axis-label font size
#     FONT_TICK = 10           # tick-label font size
#     FONT_LEG  = 8            # legend font size
#     SAVE_DPI  = 300          # resolution for raster formats (png); irrelevant for pdf/eps
#     LEG_LOC   = 'upper right'  # legend anchor inside the axes
#     LEG_NCOL  = 2            # legend columns (increase to fit more entries per row)
#     # ────────────────────────────────────────────────────────────────────────
#
#     fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
#     ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
#     ax.set_ylabel('Worst BER', fontsize=FONT_AX)
#     ax.tick_params(axis='both', labelsize=FONT_TICK)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
#     print(f"found {len(results)} models to compare")
#
#     for i, entry in enumerate(results):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         snr, ber = entry['snr'], entry['ber']
#         mask = ber > 0
#         y   = ber if mask.any() else np.full_like(ber, 1e-9)
#         lbl = entry['label'] if mask.any() else entry['label'] + '  (all zeros)'
#         ax.semilogy(snr[mask] if mask.any() else snr,
#                     y[mask]   if mask.any() else y,
#                     marker=marker, color=color, linewidth=1.5,
#                     markersize=4, label=lbl)
#
#     ylim = 1e-10
#     main_styles = [
#         dict(color='black',   linestyle='-',  marker='o'),
#         dict(color='dimgray', linestyle='--', marker='s'),
#     ]
#     for j, mb in enumerate(main_bers or []):
#         snr, ber = mb['snr'], mb['ber']
#         ber_plot = np.where(ber > 0, ber, np.nan)
#         if ~np.isnan(ber_plot).all():
#             ylim = max(ylim, ber_plot[~np.isnan(ber_plot)].min())
#         st = main_styles[j % len(main_styles)]
#         ax.semilogy(snr, ber_plot, linewidth=2.0, markersize=5,
#                     label=mb['label'], zorder=5, **st)
#
#     ax.set_ylim([ylim, 1])
#     ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
#               handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
#               columnspacing=1.0, framealpha=0.85)
#     fig.tight_layout()
#
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
#         print(f'All-models plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# def plot_variation_envelope(folder_results: list[dict], main_bers=None, save_path=None, show=True):
#     """
#     For each sub-folder plot worst/best BER envelope with filled area.
#     Single-model folders draw one line (no best/worst suffix).
#
#     folder_results: list of {
#         'label':   sub-folder name,
#         'snr':     np.ndarray  [N_snr],
#         'ber_mat': np.ndarray  [N_models, N_snr]
#     }
#
#     Paper-ready formatting:
#       • Double-column IEEE width (7 in × 4.5 in)
#       • Legend inside the axes — ncol=2 keeps it compact for two-column papers
#       • Font sizes matching a 10 pt document
#       • 300 dpi tight layout
#     To adjust: change PAPER_W/PAPER_H, FONT_* constants, or LEG_LOC/LEG_NCOL below.
#     """
#     # ── paper layout constants ── tweak these to taste ──────────────────────
#     PAPER_W   = 7.0          # figure width  in inches (3.5 = single col, 7 = double col)
#     PAPER_H   = 4.5          # figure height in inches
#     FONT_AX   = 11           # axis-label font size
#     FONT_TICK = 10           # tick-label font size
#     FONT_LEG  = 8            # legend font size
#     SAVE_DPI  = 300          # dpi for raster formats; irrelevant for pdf/eps
#     LEG_LOC   = 'upper right'  # legend anchor inside the axes
#     LEG_NCOL  = 2            # legend columns (increase if many entries)
#     # ────────────────────────────────────────────────────────────────────────
#
#     import warnings
#     fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
#     ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
#     ax.set_ylabel('Worst BER', fontsize=FONT_AX)
#     ax.tick_params(axis='both', labelsize=FONT_TICK)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
#     print(f"found {len(folder_results)} sub-folders to compare")
#
#     for i, entry in enumerate(folder_results):
#         color    = COLORS[i % len(COLORS)]
#         snr      = entry['snr']
#         mat      = entry['ber_mat']
#         n_models = mat.shape[0]
#
#         mat_safe = np.where(mat > 0, mat, np.nan)
#
#         all_nan_cols = np.all(np.isnan(mat_safe), axis=0)
#         if all_nan_cols.any():
#             bad_snr = snr[all_nan_cols]
#             print(f'  Warning [{entry["label"]}]: all models have BER=0 '
#                   f'at SNR = {bad_snr} dB — those points will be skipped.')
#
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore', category=RuntimeWarning,
#                                     message='All-NaN slice encountered')
#             worst = np.nanmax(mat_safe, axis=0)
#             best  = np.nanmin(mat_safe, axis=0)
#
#         valid = ~(np.isnan(worst) | np.isnan(best))
#
#         if n_models == 1:
#             ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
#                         marker='o', markersize=4, markevery=1,
#                         label=entry['label'])
#         else:
#             ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
#                         marker='^', markersize=4, markevery=1,
#                         label=f'{entry["label"]}  worst')
#             ax.semilogy(snr[valid], best[valid], color=color, linewidth=1.5,
#                         linestyle='--', marker='v', markersize=4, markevery=1,
#                         label=f'{entry["label"]}  best')
#             if valid.any():
#                 ax.fill_between(snr[valid], best[valid], worst[valid],
#                                 color=color, alpha=0.18)
#
#     ylim = 1e-10
#     main_styles = [
#         dict(color='black',   linestyle='-',  marker='o'),
#         dict(color='dimgray', linestyle='--', marker='s'),
#     ]
#     for j, mb in enumerate(main_bers or []):
#         snr, ber = mb['snr'], mb['ber']
#         ber_plot = np.where(ber > 0, ber, np.nan)
#         if ~np.isnan(ber_plot).all():
#             ylim = max(ylim, ber_plot[~np.isnan(ber_plot)].min())
#         st = main_styles[j % len(main_styles)]
#         ax.semilogy(snr, ber_plot, linewidth=2.0, markersize=5,
#                     label=mb['label'], zorder=5, **st)
#
#     ax.set_ylim([ylim, 1])
#     ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
#               handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
#               columnspacing=1.0, framealpha=0.85)
#     fig.tight_layout()
#
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
#         print(f'Envelope plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Architecture & Constellation helpers  (Tab 4)
# # ─────────────────────────────────────────────────────────────────────────────
#
# def int_to_binary(integers, n_bits):
#     """Convert integer tensor to binary matrix [N, n_bits] MSB-first."""
#     mask = 2 ** torch.arange(n_bits - 1, -1, -1)
#     return ((integers.unsqueeze(-1) & mask) > 0).int()
#
#
# def plot_architecture(model, folder_path, save_path=None, show=True):
#     """
#     Spatial scatter plot.
#     TX  — upward triangle  (^)  large, with black edge
#     RX  — downward triangle (v)  large, with black edge
#     Relay — circle (o)
#     Each channel gets a distinct jet color. Relays colored by dominant channel (P >= 0.5).
#     """
#     def _pt(name):
#         return torch.load(os.path.join(folder_path, 'data', name + '.pt'),
#                           weights_only=False)
#
#     posT = _pt('posT')
#     posR = _pt('posR')
#     posU = _pt('posU')
#
#     C        = model.N_channels
#     N_relays = model.N_relays
#
#     model.update_v()
#     model.culc_p()
#
#     color_RR = np.zeros((N_relays, 1))
#     for c in range(C):
#         pl = model.P[c]
#         color_RR[pl.detach().numpy() >= 0.5] = c + 1
#
#     colors = plt.get_cmap('jet', C + 1)
#     fig    = plt.figure(figsize=(12, 10))
#
#     for c in range(C):
#         col = colors(c)
#
#         plt.scatter(float(posT[c, 0]), float(posT[c, 1]),
#                     marker='^', label=f'TX - channel {c}',
#                     color=col, s=300, zorder=5,
#                     edgecolors='black', linewidths=0.6)
#
#         mask = np.squeeze(color_RR == c + 1)
#         if mask.any():
#             plt.scatter(posR[mask, 0].numpy(), posR[mask, 1].numpy(),
#                         marker='o', color=col,
#                         label=f'relay - channel - {c}', s=200, zorder=3,
#                         edgecolors='black', linewidths=0.4)
#
#         pu = posU[c]
#         pu_np = pu.numpy() if torch.is_tensor(pu) else np.array(pu)
#         if pu_np.ndim == 1:
#             pu_np = pu_np.reshape(1, -1)
#         plt.scatter(pu_np[:, 0], pu_np[:, 1],
#                     marker='v', color=col,
#                     label=f'RX - channel - {c}', s=300, zorder=5,
#                     edgecolors='black', linewidths=0.6)
#
#     plt.xticks([])
#     plt.yticks([])
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#                ncols=3 * C, mode='expand', borderaxespad=0.)
#     plt.tight_layout()
#
#     if save_path:
#         fig.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f'Architecture plot saved to {save_path}')
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# def plot_constellation(model, snr_db=10.0, n_itr=1, save_path=None, show=True):
#     """
#     One figure per channel, one figure per RX user within that channel.
#
#     Figure 1 — TX antennas (one subplot per antenna, side by side):
#         Shows the transmitted constellation with bit-string labels.
#
#     Figure 2..N_users+1 — per RX user:
#         Grid of N_symbols subplots (one per transmitted symbol).
#         Each cell shows the received cloud for that specific symbol only,
#         so overlapping symbols are split apart into separate panels.
#         Color inside each panel = transmit symbol color (consistent with TX figure).
#
#     n_itr > 1 stacks multiple noisy passes to show the full noise cloud.
#     """
#     import matplotlib.patches as mpatches
#     import math as _math
#
#     snr_lin  = dB2lin(snr_db)
#     C        = model.N_channels
#     sym_cmap = plt.get_cmap('tab20')
#
#     for c in range(C):
#         sub       = model.sub_networks[c]
#         sub.SNR   = snr_lin
#         N_users_c = int(model.N_users[c])
#         N_symbols = 2 ** N_users_c
#
#         # ── build symbol set ──────────────────────────────────────────────────
#         bits = int_to_binary(torch.arange(0, N_symbols), N_users_c).to(torch.complex64)
#         if model.demod_type == 'complex':
#             s = sub.transmitNN(bits).detach()
#         else:
#             symbols_int = torch.sum(
#                 2 ** torch.unsqueeze(
#                     torch.linspace(0, N_users_c - 1, N_users_c), dim=1)
#                 * bits.T, dim=0
#             ).real.to(torch.int32)
#             s = sub.modulation[symbols_int]   # [N_symbols] complex
#
#         print(f'Channel {c}: max TX amplitude^2 = {torch.max(s.abs() ** 2).item():.4f}')
#
#         s_plot = s.unsqueeze(1) if model.demod_type == 'simple' else s
#         if s_plot.ndim == 1:
#             s_plot = s_plot.unsqueeze(1)
#         N_tx_plot = s_plot.shape[1]
#
#         # ── forward passes ────────────────────────────────────────────────────
#         rm_runs = []
#         with torch.no_grad():
#             for _ in range(n_itr):
#                 if model.demod_type == 'complex':
#                     rm = sub(s, bits.real.T).detach().T  # [N_symbols, N_users_c]
#                 else:
#                     rm = torch.squeeze(sub(s, bits.real.T).detach(), 1).T
#
#                 rm_runs.append(rm)
#
#         bits_labels = [
#             ''.join(str(b) for b in
#                     int_to_binary(torch.tensor([sym]), N_users_c)[0].int().tolist())
#             for sym in range(N_symbols)
#         ]
#         sym_colors = [sym_cmap(sym / max(N_symbols - 1, 1)) for sym in range(N_symbols)]
#
#         def _save(fig, suffix):
#             if save_path:
#                 base, ext = os.path.splitext(save_path)
#                 ext = ext or '.png'
#                 p = f'{base}_ch{c}_{suffix}{ext}'
#                 fig.savefig(p, bbox_inches='tight', dpi=150)
#                 print(f'Saved: {p}')
#                 plt.close(fig)
#
#         # ── Figure 1: TX constellation ────────────────────────────────────────
#         fig_tx, axes_tx = plt.subplots(1, N_tx_plot,
#                                        figsize=(6 * N_tx_plot, 6),
#                                        squeeze=False)
#         fig_tx.suptitle(f'Channel {c}  |  TX  |  SNR = {snr_db:.1f} dB',
#                         fontsize=13)
#
#         for tx_idx in range(N_tx_plot):
#             ax = axes_tx[0, tx_idx]
#             for sym in range(N_symbols):
#                 ax.scatter(s_plot[sym, tx_idx].real.item(),
#                            s_plot[sym, tx_idx].imag.item(),
#                            color=sym_colors[sym], s=100, zorder=3)
#                 ax.annotate(bits_labels[sym],
#                             (s_plot[sym, tx_idx].real.item(),
#                              s_plot[sym, tx_idx].imag.item()),
#                             textcoords='offset points', xytext=(6, 6),
#                             fontsize=9, fontweight='bold')
#             ax.set_title(f'TX antenna {tx_idx}', fontsize=11)
#             ax.set_xlabel('Real');  ax.set_ylabel('Imag')
#             ax.grid(True, linestyle='--', alpha=0.4)
#             ax.axhline(0, color='gray', lw=0.5)
#             ax.axvline(0, color='gray', lw=0.5)
#             handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
#                        for s in range(N_symbols)]
#             ax.legend(handles=handles, title='Symbol', fontsize=7,
#                       ncol=max(1, N_symbols // 4), loc='best')
#
#         fig_tx.tight_layout()
#         _save(fig_tx, 'TX')
#
#         # ── One figure per RX user — all symbols on a single plot ──────────
#         for u in range(N_users_c):
#             fig_rx, ax = plt.subplots(figsize=(7, 7))
#             fig_rx.suptitle(
#                 f'Channel {c}  |  RX user {u}  |  SNR = {snr_db:.1f} dB  '
#                 f'({n_itr} iteration{"s" if n_itr > 1 else ""})',
#                 fontsize=13)
#
#             for sym in range(N_symbols):
#                 pts = torch.stack([rm_runs[it][sym, u] for it in range(n_itr)])
#                 ax.scatter(pts.real.numpy(), pts.imag.numpy(),
#                            color=sym_colors[sym], s=20, alpha=0.6,
#                            zorder=3, label=bits_labels[sym])
#                 # centroid marker
#                 cx = pts.real.mean().item()
#                 cy = pts.imag.mean().item()
#                 ax.scatter([cx], [cy], color=sym_colors[sym], s=80,
#                            marker='+', zorder=5, linewidths=2)
#                 ax.annotate(bits_labels[sym], (cx, cy),
#                             textcoords='offset points', xytext=(6, 6),
#                             fontsize=9, fontweight='bold',
#                             color=sym_colors[sym])
#
#             ax.set_xlabel('Real', fontsize=10)
#             ax.set_ylabel('Imag', fontsize=10)
#             ax.grid(True, linestyle='--', alpha=0.4)
#             ax.axhline(0, color='gray', lw=0.5)
#             ax.axvline(0, color='gray', lw=0.5)
#             handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
#                        for s in range(N_symbols)]
#             ax.legend(handles=handles, title='Symbol', fontsize=8,
#                       ncol=max(1, N_symbols // 4), loc='best')
#
#             fig_rx.tight_layout()
#             _save(fig_rx, f'RX_user{u}')
#     if show:
#         plt.show()
#
#
#
#
# # GUI
# # ─────────────────────────────────────────────────────────────────────────────
#
# class App(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title('Model Comparison Tool')
#         self.resizable(True, True)
#         self._ber_rows: list[dict] = []
#         self._log_rows: list[dict] = []
#         self._build_ui()
#
#     # ── top-level notebook ────────────────────────────────────────────────────
#
#     def _build_ui(self):
#         nb = ttk.Notebook(self)
#         nb.pack(fill='both', expand=True, padx=6, pady=6)
#
#         self._tab_ber  = tk.Frame(nb)
#         self._tab_log  = tk.Frame(nb)
#         self._tab_var  = tk.Frame(nb)
#         self._tab_arch = tk.Frame(nb)
#         nb.add(self._tab_ber,  text='  BER Comparison  ')
#         nb.add(self._tab_log,  text='  Log Comparison  ')
#         nb.add(self._tab_var,  text='  Variation Comparison  ')
#         nb.add(self._tab_arch, text='  Architecture & Constellation  ')
#
#         self._build_ber_tab(self._tab_ber)
#         self._build_log_tab(self._tab_log)
#         self._build_var_tab(self._tab_var)
#         self._build_arch_tab(self._tab_arch)
#
#     # ══════════════════════════════════════════════════════════════════════════
#     # TAB 1 — BER
#     # ══════════════════════════════════════════════════════════════════════════
#
#     def _build_ber_tab(self, parent):
#         pad = dict(padx=8, pady=4)
#
#         lf_models = tk.LabelFrame(parent, text='Models', **pad)
#         lf_models.grid(row=0, column=0, columnspan=2, sticky='nsew', **pad)
#         self._ber_inner = tk.Frame(lf_models)
#         self._ber_inner.pack(fill='both', expand=True)
#         tk.Button(lf_models, text='＋  Add model folder',
#                   command=self._ber_add).pack(anchor='w', **pad)
#
#         lf_eval = tk.LabelFrame(parent, text='Evaluation settings', **pad)
#         lf_eval.grid(row=1, column=0, sticky='nsew', **pad)
#
#         def _erow(label, default, r):
#             tk.Label(lf_eval, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
#             v = tk.StringVar(value=default)
#             tk.Entry(lf_eval, textvariable=v, width=10).grid(row=r, column=1, **pad)
#             return v
#
#         self._snr_min  = _erow('SNR min (dB)',   '-20',  0)
#         self._snr_max  = _erow('SNR max (dB)',   '40',   1)
#         self._snr_step = _erow('SNR step (dB)',  '1',    2)
#         self._batch    = _erow('Batch size',     '1000', 3)
#         self._n_itr    = _erow('Avg iterations', '10',   4)
#
#         lf_out = tk.LabelFrame(parent, text='Output', **pad)
#         lf_out.grid(row=1, column=1, sticky='nsew', **pad)
#         tk.Label(lf_out, text='Save plot to (optional)', anchor='w').grid(
#             row=0, column=0, sticky='w', **pad)
#         self._ber_save = tk.StringVar()
#         tk.Entry(lf_out, textvariable=self._ber_save, width=30).grid(row=0, column=1, **pad)
#         tk.Button(lf_out, text='Browse…',
#                   command=lambda: self._browse_save(self._ber_save)).grid(row=0, column=2, **pad)
#         self._ber_show = tk.BooleanVar(value=True)
#         tk.Checkbutton(lf_out, text='Show plot interactively',
#                        variable=self._ber_show).grid(row=1, column=0, columnspan=3, sticky='w', **pad)
#
#         tk.Button(parent, text='▶  Run BER comparison', font=('', 11, 'bold'),
#                   bg='#2E86AB', fg='white',
#                   command=self._ber_run).grid(row=2, column=0, columnspan=2,
#                                               sticky='ew', padx=8, pady=8)
#         parent.columnconfigure(0, weight=1)
#         parent.columnconfigure(1, weight=1)
#
#     def _ber_add(self, path=''):
#         if not path:
#             path = filedialog.askdirectory(title='Select model folder')
#         if not path:
#             return
#         frame = tk.Frame(self._ber_inner, relief='groove', bd=1, padx=4, pady=4)
#         frame.pack(fill='x', pady=2)
#         display = path if len(path) <= 55 else '…' + path[-52:]
#         tk.Label(frame, text=display, anchor='w', width=52).grid(row=0, column=0, sticky='w')
#
#         # one checkbox per stage — multiple can be ticked simultaneously
#         tk.Label(frame, text='Stages:').grid(row=0, column=1, padx=(8, 2))
#         stage_vars = {}
#         for col, s in enumerate((1, 2, 3), start=2):
#             var = tk.BooleanVar(value=(s == 3))   # stage 3 ticked by default
#             tk.Checkbutton(frame, text=str(s), variable=var).grid(row=0, column=col, padx=2)
#             stage_vars[s] = var
#
#         tk.Label(frame, text='Label:').grid(row=0, column=5, padx=(8, 2))
#         label_var = tk.StringVar(value=os.path.basename(path.rstrip('/\\')))
#         tk.Entry(frame, textvariable=label_var, width=28).grid(row=0, column=6, padx=2)
#
#         def _remove(f=frame):
#             f.destroy()
#             self._ber_rows = [r for r in self._ber_rows if r['frame'] is not f]
#
#         tk.Button(frame, text='✕', fg='red', command=_remove).grid(row=0, column=7, padx=(8, 0))
#         self._ber_rows.append(dict(path=path, stage_vars=stage_vars,
#                                    label_var=label_var, frame=frame))
#         # auto-set SNR min from this model if it's the first one added
#         if len(self._ber_rows) == 1:
#             max_snr = read_max_snr_stage1(path)
#             if max_snr is not None:
#                 self._snr_min.set(str(int(round(max_snr - 10))))
#
#     def _ber_run(self):
#         rows = [r for r in self._ber_rows if r['frame'].winfo_exists()]
#         if not rows:
#             messagebox.showwarning('No models', 'Add at least one model folder.')
#             return
#         try:
#             snr_min  = float(self._snr_min.get())
#             snr_max  = float(self._snr_max.get())
#             snr_step = float(self._snr_step.get())
#             batch    = int(self._batch.get())
#             num_itr  = int(self._n_itr.get())
#             snr_range = torch.arange(snr_min, snr_max + snr_step * 0.5, snr_step)
#         except ValueError as e:
#             messagebox.showerror('Invalid setting', str(e)); return
#
#         save_path = self._ber_save.get().strip() or None
#         results   = []
#         for row in rows:
#             path  = row['path']
#             name  = row['label_var'].get().strip() or os.path.basename(path.rstrip('/\\'))
#             selected_stages = [s for s, var in row['stage_vars'].items() if var.get()]
#             if not selected_stages:
#                 messagebox.showwarning('No stage selected',
#                                        f'Select at least one stage for:\n{name}')
#                 continue
#             for stage in selected_stages:
#                 print(f'\n{"="*55}\n  Loading: {name}  |  stage {stage}\n{"="*55}')
#                 try:
#                     model = load_model(path, stage)
#                 except Exception as e:
#                     messagebox.showerror('Load error',
#                                          f'Failed to load stage {stage}:\n{path}\n\n{e}')
#                     continue
#                 ber = evaluate_model(model, snr_range, batch, num_itr)
#                 results.append(dict(name=name, stage=stage, ber=ber))
#                 if save_path:
#                     base = os.path.splitext(save_path)[0]
#                     safe = name.replace(' ', '_').replace('\\', '_').replace('/', '_')
#                     torch.save(ber, f'{base}_{safe}_stage{stage}_BER.pt')
#         if not results:
#             messagebox.showinfo('Done', 'No models evaluated.'); return
#         plot_ber_comparison(results, snr_range,
#                             save_path=save_path, show=self._ber_show.get())
#
#     def _build_log_tab(self, parent):
#         pad = dict(padx=8, pady=4)
#
#         # ── top: log file list ────────────────────────────────────────────────
#         lf_logs = tk.LabelFrame(parent, text='Log files', **pad)
#         lf_logs.grid(row=0, column=0, columnspan=3, sticky='nsew', **pad)
#
#         self._log_inner = tk.Frame(lf_logs)
#         self._log_inner.pack(fill='both', expand=True)
#
#         btn_row = tk.Frame(lf_logs)
#         btn_row.pack(fill='x', pady=(4, 0))
#         tk.Button(btn_row, text='＋  Add log file (.pt)',
#                   command=self._log_add_file).pack(side='left', **pad)
#         tk.Button(btn_row, text='＋  Add model folder  (auto-finds log)',
#                   command=self._log_add_folder).pack(side='left', **pad)
#
#         # ── middle: metric selectors ──────────────────────────────────────────
#         lf_metric = tk.LabelFrame(parent, text='Metrics to plot', **pad)
#         lf_metric.grid(row=1, column=0, sticky='nsew', **pad)
#
#         # Left axis (always shown)
#         tk.Label(lf_metric, text='Left axis (Y1):', anchor='w',
#                  font=('', 9, 'bold')).grid(row=0, column=0, sticky='w', **pad)
#         self._metric_var = tk.StringVar()
#         self._metric_cb  = ttk.Combobox(lf_metric, textvariable=self._metric_var,
#                                          state='readonly', width=36)
#         self._metric_cb.grid(row=0, column=1, sticky='w', **pad)
#
#         # Right axis (only active when one log is loaded)
#         self._metric2_label = tk.Label(lf_metric, text='Right axis (Y2):', anchor='w',
#                                         font=('', 9, 'bold'))
#         self._metric2_label.grid(row=1, column=0, sticky='w', **pad)
#         self._metric2_var = tk.StringVar()
#         self._metric2_cb  = ttk.Combobox(lf_metric, textvariable=self._metric2_var,
#                                           state='readonly', width=36)
#         self._metric2_cb.grid(row=1, column=1, sticky='w', **pad)
#         self._metric2_none_lbl = tk.Label(lf_metric,
#                                            text='(available when only one log is loaded)',
#                                            fg='gray', font=('', 8, 'italic'))
#         self._metric2_none_lbl.grid(row=1, column=2, sticky='w', padx=4)
#
#         tk.Button(lf_metric, text='↺  Refresh',
#                   command=self._log_refresh_metrics).grid(row=0, column=2, **pad)
#
#         # Moving-average window
#         tk.Label(lf_metric, text='MA window:').grid(row=2, column=0, sticky='w', **pad)
#         self._ma_var = tk.StringVar(value='10')
#         tk.Entry(lf_metric, textvariable=self._ma_var, width=6).grid(row=2, column=1,
#                                                                        sticky='w', **pad)
#
#         # ── right: output ─────────────────────────────────────────────────────
#         lf_out = tk.LabelFrame(parent, text='Output', **pad)
#         lf_out.grid(row=1, column=1, sticky='nsew', **pad)
#         tk.Label(lf_out, text='Save plot to (optional)', anchor='w').grid(
#             row=0, column=0, sticky='w', **pad)
#         self._log_save = tk.StringVar()
#         tk.Entry(lf_out, textvariable=self._log_save, width=28).grid(row=0, column=1, **pad)
#         tk.Button(lf_out, text='Browse…',
#                   command=lambda: self._browse_save(self._log_save)).grid(row=0, column=2, **pad)
#         self._log_show = tk.BooleanVar(value=True)
#         tk.Checkbutton(lf_out, text='Show plot interactively',
#                        variable=self._log_show).grid(row=1, column=0,
#                                                      columnspan=3, sticky='w', **pad)
#
#         # ── run button ────────────────────────────────────────────────────────
#         tk.Button(parent, text='▶  Plot logs', font=('', 11, 'bold'),
#                   bg='#3D8A40', fg='white',
#                   command=self._log_run).grid(row=2, column=0, columnspan=3,
#                                               sticky='ew', padx=8, pady=8)
#
#         parent.columnconfigure(0, weight=1)
#         parent.columnconfigure(1, weight=1)
#
#     # ── log row management ────────────────────────────────────────────────────
#
#     def _log_add_file(self, filepath='', label_default=''):
#         """Add a row for a .pt log file."""
#         if not filepath:
#             filepath = filedialog.askopenfilename(
#                 title='Select log .pt file',
#                 filetypes=[('PyTorch files', '*.pt'), ('All files', '*.*')])
#         if not filepath:
#             return
#         try:
#             test = torch.load(filepath, weights_only=False)
#             if 'log' in test:
#                 test = test['log']
#             assert 'itr_axis' in test, 'No itr_axis key — not a valid log file'
#         except Exception as e:
#             messagebox.showerror('Invalid log file', str(e)); return
#
#         frame = tk.Frame(self._log_inner, relief='groove', bd=1, padx=4, pady=4)
#         frame.pack(fill='x', pady=2)
#
#         display = filepath if len(filepath) <= 60 else '…' + filepath[-57:]
#         tk.Label(frame, text=display, anchor='w', width=62,
#                  font=('', 8)).grid(row=0, column=0, sticky='w')
#         tk.Label(frame, text='Label:').grid(row=0, column=1, padx=(8, 2))
#         default_label = label_default or os.path.basename(
#             os.path.dirname(filepath)).replace('_', ' ')
#         label_var = tk.StringVar(value=default_label)
#         tk.Entry(frame, textvariable=label_var, width=22).grid(row=0, column=2, padx=2)
#
#         self._log_rows.append(dict(filepath=filepath, label_var=label_var, frame=frame))
#
#         def _remove(f=frame):
#             f.destroy()
#             self._log_rows = [r for r in self._log_rows if r['frame'] is not f]
#             self._log_refresh_metrics()
#
#         tk.Button(frame, text='✕', fg='red',
#                   command=_remove).grid(row=0, column=3, padx=(8, 0))
#
#         self._log_refresh_metrics()
#
#     def _log_add_folder(self):
#         """Auto-find stage2_all_epochs_raw.pt inside a model folder."""
#         folder = filedialog.askdirectory(title='Select model folder')
#         if not folder:
#             return
#         candidates = [
#             os.path.join(folder, 'data', 'stage2_all_epochs_raw.pt'),
#             os.path.join(folder, 'stage2_all_epochs_raw.pt'),
#         ]
#         found = next((p for p in candidates if os.path.exists(p)), None)
#         if found is None:
#             found = filedialog.askopenfilename(
#                 initialdir=os.path.join(folder, 'data'),
#                 title=f'Cannot auto-find log in {folder} — select manually',
#                 filetypes=[('PyTorch files', '*.pt'), ('All files', '*.*')])
#         if found:
#             self._log_add_file(filepath=found,
#                                label_default=os.path.basename(folder.rstrip('/\\')))
#
#     def _log_refresh_metrics(self):
#         """Refresh both metric dropdowns from the first loaded log."""
#         valid = [r for r in self._log_rows if r['frame'].winfo_exists()]
#
#         if not valid:
#             self._metric_cb['values']  = []
#             self._metric2_cb['values'] = []
#             return
#
#         try:
#             log  = load_log(valid[0]['filepath'])
#             opts = expand_metric_names(log)
#             self._metric_map = {display: (k, ch) for display, k, ch in opts}
#             names = list(self._metric_map.keys())
#
#             self._metric_cb['values']  = names
#             self._metric2_cb['values'] = ['(none)'] + names
#
#             if self._metric_var.get() not in self._metric_map:
#                 self._metric_cb.current(0)
#             if self._metric2_var.get() not in self._metric_map:
#                 self._metric2_cb.current(0)   # sets to '(none)'
#         except Exception as e:
#             messagebox.showerror('Metric refresh error', str(e))
#             return
#
#         # Enable / disable second metric depending on number of logs
#         single = (len(valid) == 1)
#         state  = 'readonly' if single else 'disabled'
#         self._metric2_cb.config(state=state)
#         self._metric2_none_lbl.config(
#             text='' if single else '(available when only one log is loaded)')
#
#     # ── run log plot ──────────────────────────────────────────────────────────
#
#     def _log_run(self):
#         valid = [r for r in self._log_rows if r['frame'].winfo_exists()]
#         if not valid:
#             messagebox.showwarning('No logs', 'Add at least one log file.'); return
#
#         if not self._metric_var.get():
#             messagebox.showwarning('No metric', 'Select a metric to plot.'); return
#
#         if not hasattr(self, '_metric_map'):
#             self._log_refresh_metrics()
#
#         m1_display = self._metric_var.get()
#         m2_display = self._metric2_var.get()
#         if m1_display not in self._metric_map:
#             messagebox.showerror('Error', f'Unknown metric: {m1_display}'); return
#
#         m1_key, m1_ch = self._metric_map[m1_display]
#         m2_key, m2_ch = None, None
#         use_twin = (len(valid) == 1
#                     and m2_display
#                     and m2_display != '(none)'
#                     and m2_display in self._metric_map)
#         if use_twin:
#             m2_key, m2_ch = self._metric_map[m2_display]
#
#         try:
#             window = int(self._ma_var.get())
#         except ValueError:
#             window = 10
#
#         # ── load all logs ─────────────────────────────────────────────────────
#         entries = []
#         for row in valid:
#             label = row['label_var'].get().strip()
#             try:
#                 log = load_log(row['filepath'])
#             except Exception as e:
#                 messagebox.showerror('Load error', f'{label}:\n{e}'); continue
#             entries.append(dict(label=label, log=log))
#
#         if not entries:
#             return
#
#         save_path = self._log_save.get().strip() or None
#
#         if use_twin:
#             # single log, two metrics, twin y-axes
#             _plot_dual_metric(entries[0], m1_key, m1_ch, m2_key, m2_ch,
#                               window=window, save_path=save_path,
#                               show=self._log_show.get())
#         else:
#             # multiple logs, one metric
#             plot_log_comparison(entries, m1_key, m1_ch,
#                                 window=window, save_path=save_path,
#                                 show=self._log_show.get())
#
#     # ══════════════════════════════════════════════════════════════════════════
#     # TAB 3 — Variation Comparison
#     # ══════════════════════════════════════════════════════════════════════════
#
#     def _build_var_tab(self, parent):
#         pad = dict(padx=8, pady=4)
#
#         # ── main folder selector ──────────────────────────────────────────────
#         lf_folder = tk.LabelFrame(parent, text='Main model folder', **pad)
#         lf_folder.grid(row=0, column=0, columnspan=2, sticky='ew', **pad)
#         lf_folder.columnconfigure(1, weight=1)
#
#         tk.Label(lf_folder, text='Folder:').grid(row=0, column=0, sticky='w', **pad)
#         self._var_folder_var = tk.StringVar()
#         self._var_folder_entry = tk.Entry(lf_folder, textvariable=self._var_folder_var,
#                                           width=60, state='readonly')
#         self._var_folder_entry.grid(row=0, column=1, sticky='ew', **pad)
#         tk.Button(lf_folder, text='Browse…',
#                   command=self._var_browse).grid(row=0, column=2, **pad)
#         tk.Button(lf_folder, text='⟳  Scan',
#                   command=self._var_scan).grid(row=0, column=3, **pad)
#
#         tk.Label(lf_folder, text='Main model label:').grid(row=1, column=0, sticky='w', **pad)
#         self._var_main_label = tk.StringVar(value='main model')
#         tk.Entry(lf_folder, textvariable=self._var_main_label,
#                  width=40).grid(row=1, column=1, sticky='w', **pad)
#
#         # ── sub-folder checklist ──────────────────────────────────────────────
#         lf_sf = tk.LabelFrame(parent, text='Sub-folders  (select which to include)', **pad)
#         lf_sf.grid(row=1, column=0, columnspan=2, sticky='nsew', **pad)
#         lf_sf.columnconfigure(0, weight=1)
#         lf_sf.rowconfigure(0, weight=1)
#
#         sf_canvas = tk.Canvas(lf_sf, height=160, highlightthickness=0)
#         sf_vsb = tk.Scrollbar(lf_sf, orient='vertical', command=sf_canvas.yview)
#         sf_canvas.configure(yscrollcommand=sf_vsb.set)
#         sf_vsb.grid(row=0, column=1, sticky='ns')
#         sf_canvas.grid(row=0, column=0, sticky='nsew')
#
#         self._sf_inner = tk.Frame(sf_canvas)
#         sf_inner_id = sf_canvas.create_window((0, 0), window=self._sf_inner, anchor='nw')
#
#         def _sf_frame_cfg(e):
#             sf_canvas.configure(scrollregion=sf_canvas.bbox('all'))
#         self._sf_inner.bind('<Configure>', _sf_frame_cfg)
#
#         def _sf_canvas_cfg(e):
#             sf_canvas.itemconfig(sf_inner_id, width=e.width)
#         sf_canvas.bind('<Configure>', _sf_canvas_cfg)
#         self._sf_canvas = sf_canvas
#         self._sf_entries: list[dict] = []   # {name, enabled_var, frame}
#
#         sf_btn_row = tk.Frame(lf_sf)
#         sf_btn_row.grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 0))
#         tk.Button(sf_btn_row, text='Select all',
#                   command=self._var_sf_select_all).pack(side='left', padx=4)
#         tk.Button(sf_btn_row, text='Select none',
#                   command=self._var_sf_select_none).pack(side='left', padx=4)
#
#         # ── SNR range ─────────────────────────────────────────────────────────
#         lf_snr = tk.LabelFrame(parent, text='SNR range  (integer steps of 1)', **pad)
#         lf_snr.grid(row=2, column=0, sticky='nsew', **pad)
#
#         def _srow(label, default, r):
#             tk.Label(lf_snr, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
#             v = tk.StringVar(value=default)
#             tk.Entry(lf_snr, textvariable=v, width=8).grid(row=r, column=1, **pad)
#             return v
#
#         self._var_snr_min  = _srow('SNR min (dB)',  '-20', 0)
#         self._var_snr_max  = _srow('SNR max (dB)',  '40', 1)
#         self._var_snr_step = _srow('SNR step (dB)', '1',  2)
#
#         # ── output ────────────────────────────────────────────────────────────
#         lf_out = tk.LabelFrame(parent, text='Output', **pad)
#         lf_out.grid(row=2, column=1, sticky='nsew', **pad)
#         tk.Label(lf_out, text='Save plot to (optional)', anchor='w').grid(
#             row=0, column=0, sticky='w', **pad)
#         self._var_save = tk.StringVar()
#         tk.Entry(lf_out, textvariable=self._var_save, width=28).grid(row=0, column=1, **pad)
#         tk.Button(lf_out, text='Browse…',
#                   command=lambda: self._browse_save(self._var_save)).grid(row=0, column=2, **pad)
#         self._var_show = tk.BooleanVar(value=True)
#         tk.Checkbutton(lf_out, text='Show plot interactively',
#                        variable=self._var_show).grid(row=1, column=0,
#                                                      columnspan=3, sticky='w', **pad)
#
#         # ── run buttons ───────────────────────────────────────────────────────
#         btn_frame = tk.Frame(parent)
#         btn_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
#         btn_frame.columnconfigure(0, weight=1)
#         btn_frame.columnconfigure(1, weight=1)
#
#         tk.Button(btn_frame, text='▶  Plot all models', font=('', 11, 'bold'),
#                   bg='#27AE60', fg='white',
#                   command=self._var_run_all).grid(row=0, column=0, sticky='ew', padx=(0, 4))
#         tk.Button(btn_frame, text='▶  Plot BER envelope', font=('', 11, 'bold'),
#                   bg='#2E86AB', fg='white',
#                   command=self._var_run_envelope).grid(row=0, column=1, sticky='ew', padx=(4, 0))
#
#         parent.columnconfigure(0, weight=1)
#         parent.columnconfigure(1, weight=1)
#         parent.rowconfigure(1, weight=1)
#
#     # ── variation helpers ─────────────────────────────────────────────────────
#
#     def _var_browse(self):
#         path = filedialog.askdirectory(title='Select main model folder')
#         if path:
#             self._var_folder_entry.configure(state='normal')
#             self._var_folder_var.set(path)
#             self._var_folder_entry.configure(state='readonly')
#             self._var_scan()
#
#     def _var_scan(self):
#         """Scan main folder and populate the sub-folder checklist."""
#         folder = self._var_folder_var.get().strip()
#         if not folder:
#             messagebox.showwarning('No folder', 'Please select a main model folder first.')
#             return
#         # auto-fill the main model label with the folder name (user can override)
#         self._var_main_label.set(os.path.basename(folder.rstrip('/\\')))
#
#         for w in self._sf_inner.winfo_children():
#             w.destroy()
#         self._sf_entries.clear()
#
#         # auto-set SNR min from the main model's data folder (always, even if no subfolders)
#         max_snr = read_max_snr_stage1(folder)
#         if max_snr is not None:
#             self._var_snr_min.set(str(int(round(max_snr - 10))))
#             print(f'  Auto SNR min set to {int(round(max_snr - 10))} dB '
#                   f'(max_snr_train_stage_1={max_snr:.1f} - 10)')
#
#         subfolders = discover_subfolders(folder)
#         if not subfolders:
#             tk.Label(self._sf_inner,
#                      text='No sub-folders with valid models found.\n'
#                           'Expected: <main>/<sub_folder>/<model>/outputs/{SNR.pt, worst_BER.pt}',
#                      fg='gray', justify='left').pack(anchor='w', padx=8, pady=8)
#             self._sf_canvas.configure(scrollregion=self._sf_canvas.bbox('all'))
#             return
#
#         for sf_name in subfolders:
#             frame = tk.Frame(self._sf_inner, relief='flat', bd=0)
#             frame.pack(fill='x', pady=1, padx=4)
#             enabled_var = tk.BooleanVar(value=True)
#             tk.Checkbutton(frame, variable=enabled_var).grid(row=0, column=0)
#             display_var = tk.StringVar(value=sf_name)
#             tk.Entry(frame, textvariable=display_var, width=40,
#                      font=('', 9)).grid(row=0, column=1, padx=4, sticky='w')
#             tk.Label(frame, text=sf_name, fg='gray',
#                      font=('', 7)).grid(row=0, column=2, padx=(2, 8), sticky='w')
#             self._sf_entries.append({'name': sf_name, 'display_var': display_var,
#                                      'enabled_var': enabled_var, 'frame': frame})
#
#         self._sf_inner.update_idletasks()
#         self._sf_canvas.configure(scrollregion=self._sf_canvas.bbox('all'))
#         print(f'Found {len(subfolders)} sub-folder(s).')
#
#     def _var_sf_select_all(self):
#         for e in self._sf_entries:
#             e['enabled_var'].set(True)
#
#     def _var_sf_select_none(self):
#         for e in self._sf_entries:
#             e['enabled_var'].set(False)
#
#     def _var_get_snr_and_active_sf(self):
#         """Shared validation: returns (snr_targets, active_sf_names) or (None, None)."""
#         try:
#             snr_min  = float(self._var_snr_min.get())
#             snr_max  = float(self._var_snr_max.get())
#             snr_step = float(self._var_snr_step.get())
#         except ValueError as e:
#             messagebox.showerror('Invalid SNR', str(e))
#             return None, None
#
#         if snr_min >= snr_max:
#             messagebox.showerror('Invalid SNR', 'SNR min must be less than SNR max.')
#             return None, None
#         if snr_step <= 0:
#             messagebox.showerror('Invalid SNR', 'SNR step must be positive.')
#             return None, None
#
#         active_sf = [(e['name'], e['display_var'].get().strip() or e['name'])
#                      for e in self._sf_entries
#                      if e['frame'].winfo_exists() and e['enabled_var'].get()]
#         if not active_sf:
#             messagebox.showwarning('Nothing selected', 'Please select at least one sub-folder.')
#             return None, None
#
#         return np.arange(snr_min, snr_max + snr_step * 0.5, snr_step), active_sf
#
#     def _var_load_main_ber(self, snr_targets):
#         """
#         Load stage-3 and stage-1 worst BER from <main_folder>/outputs/.
#         Files expected:
#             SNR.pt
#             worst_BER_stage_3.pt   — shape [N_snr] or [N_channels, N_snr]
#             worst_BER_stage_1.pt   — same shape (optional)
#         Returns a list of dicts (one per stage found), each:
#             {'label', 'snr': np.ndarray, 'ber': np.ndarray}
#         Returns empty list if the outputs folder / SNR file is missing.
#         """
#         folder = self._var_folder_var.get().strip()
#         out    = os.path.join(folder, 'outputs')
#         snr_f  = os.path.join(out, 'SNR.pt')
#         base_label = self._var_main_label.get().strip() or os.path.basename(folder.rstrip('/\\'))
#
#         print(f'  Looking for main model outputs in: {out}')
#
#         if not os.path.isfile(snr_f):
#             messagebox.showinfo('Main model',
#                                 f'SNR.pt not found at:\n{snr_f}\n\n'
#                                 'Main model will not be plotted.')
#             return []
#
#         try:
#             snr_raw = torch.load(snr_f, weights_only=True)
#             snr_np  = (snr_raw.numpy() if torch.is_tensor(snr_raw)
#                        else np.array(snr_raw)).flatten()
#         except Exception as ex:
#             messagebox.showwarning('Main model SNR load error', str(ex))
#             return []
#
#         def _load_stage_ber(stage):
#             ber_f = os.path.join(out, f'worst_BER_stage_{stage}.pt')
#             print(f'    worst_BER_stage_{stage}.pt exists: {os.path.isfile(ber_f)}')
#             if not os.path.isfile(ber_f):
#                 return None
#             try:
#                 ber_raw = torch.load(ber_f, weights_only=True)
#                 ber_np  = (ber_raw.numpy() if torch.is_tensor(ber_raw)
#                            else np.array(ber_raw))
#                 if ber_np.ndim == 2:
#                     ber_np = ber_np.max(axis=0)
#                 ber_np = ber_np.flatten()
#
#                 matched_snr, matched_ber = [], []
#                 for t in snr_targets:
#                     idx = int(np.argmin(np.abs(snr_np - t)))
#                     matched_snr.append(snr_np[idx])
#                     matched_ber.append(ber_np[idx])
#                 snr_m = np.array(matched_snr)
#                 ber_m = np.array(matched_ber)
#                 print(f'  Main stage {stage} loaded. BER range: {ber_m.min():.2e}–{ber_m.max():.2e}')
#                 return {'label': f'{base_label}  (main stage {stage})',
#                         'snr': snr_m, 'ber': ber_m}
#             except Exception as ex:
#                 print(f'  Warning: could not load stage {stage} BER: {ex}')
#                 return None
#
#         results = []
#         for stage in (3, 1):
#             entry = _load_stage_ber(stage)
#             if entry is not None:
#                 results.append(entry)
#
#         if not results:
#             messagebox.showinfo('Main model',
#                                 f'No worst_BER_stage_*.pt files found in:\n{out}\n\n'
#                                 'Main model will not be plotted.')
#         return results
#
#     def _var_run_all(self):
#         """Plot every individual model curve."""
#         snr_targets, active_sf = self._var_get_snr_and_active_sf()
#         if snr_targets is None:
#             return
#
#         folder   = self._var_folder_var.get().strip()
#         sf_names = [name for name, _ in active_sf]
#         sf_labels = {name: lbl for name, lbl in active_sf}
#         models = discover_models_in_subfolders(folder, sf_names)
#         if not models:
#             messagebox.showinfo('No models', 'No valid models found in selected sub-folders.')
#             return
#
#         results, errors = [], []
#         for m in models:
#             # replace the sf_name part of the label with the user's display label
#             sf_part   = m['label'].split(' / ')[0]
#             model_part = m['label'].split(' / ', 1)[1] if ' / ' in m['label'] else m['label']
#             display_label = f"{sf_labels.get(sf_part, sf_part)} / {model_part}"
#             try:
#                 snr_m, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
#                 results.append({'label': display_label, 'snr': snr_m, 'ber': ber_m})
#             except Exception as ex:
#                 errors.append(f"{display_label}:\n  {ex}")
#
#         if errors:
#             messagebox.showwarning('Load warnings',
#                                    'Some models could not be loaded:\n\n' + '\n\n'.join(errors))
#         if not results:
#             messagebox.showinfo('Done', 'No models loaded successfully.')
#             return
#
#         save_path = self._var_save.get().strip() or None
#         main_bers = self._var_load_main_ber(snr_targets)
#         plot_variation_all(results, main_bers=main_bers, save_path=save_path, show=self._var_show.get())
#
#     def _var_run_envelope(self):
#         """For each sub-folder plot the worst/best BER envelope with filled area."""
#         snr_targets, active_sf = self._var_get_snr_and_active_sf()
#         if snr_targets is None:
#             return
#
#         folder = self._var_folder_var.get().strip()
#         folder_results, errors = [], []
#
#         for sf_name, sf_display in active_sf:
#             models = discover_models_in_subfolders(folder, [sf_name])
#             if not models:
#                 errors.append(f'{sf_name}: no valid models found')
#                 continue
#
#             ber_rows = []
#             for m in models:
#                 try:
#                     _, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
#                     ber_rows.append(ber_m)
#                 except Exception as ex:
#                     errors.append(f"{m['label']}:\n  {ex}")
#
#             if not ber_rows:
#                 continue
#
#             ber_mat = np.vstack(ber_rows)          # [N_models, N_snr]
#             snr_rep, _ = load_variation_ber(models[0]['outputs_path'], snr_targets)
#
#             folder_results.append({
#                 'label':   sf_display,
#                 'snr':     snr_rep,
#                 'ber_mat': ber_mat,
#             })
#             print(f'  {sf_display} ({sf_name}): {len(ber_rows)} model(s) loaded')
#
#         if errors:
#             messagebox.showwarning('Load warnings',
#                                    'Some models could not be loaded:\n\n' + '\n\n'.join(errors))
#         if not folder_results:
#             messagebox.showinfo('Done', 'No data to plot.')
#             return
#
#         save_path = self._var_save.get().strip() or None
#         main_bers = self._var_load_main_ber(snr_targets)
#         plot_variation_envelope(folder_results, main_bers=main_bers, save_path=save_path, show=self._var_show.get())
#
#             # ── shared helpers ────────────────────────────────────────────────────────
#
#     def _browse_save(self, var):
#         p = filedialog.asksaveasfilename(
#             defaultextension='.png',
#             filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'), ('All files', '*.*')],
#             title='Save plot as…')
#         if p:
#             var.set(p)
#
#
#     # ══════════════════════════════════════════════════════════════════════════
#     # TAB 4 — Architecture & Constellation
#     # ══════════════════════════════════════════════════════════════════════════
#
#     def _build_arch_tab(self, parent):
#         pad = dict(padx=8, pady=4)
#
#         # ── model selector ────────────────────────────────────────────────────
#         lf_model = tk.LabelFrame(parent, text='Model', **pad)
#         lf_model.grid(row=0, column=0, columnspan=2, sticky='ew', **pad)
#         lf_model.columnconfigure(1, weight=1)
#
#         tk.Label(lf_model, text='Folder:').grid(row=0, column=0, sticky='w', **pad)
#         self._arch_folder_var = tk.StringVar()
#         self._arch_folder_entry = tk.Entry(lf_model, textvariable=self._arch_folder_var,
#                                            width=60, state='readonly')
#         self._arch_folder_entry.grid(row=0, column=1, sticky='ew', **pad)
#         tk.Button(lf_model, text='Browse…',
#                   command=self._arch_browse).grid(row=0, column=2, **pad)
#
#         tk.Label(lf_model, text='Stage:').grid(row=1, column=0, sticky='w', **pad)
#         self._arch_stage_var = tk.IntVar(value=3)
#         sf = tk.Frame(lf_model)
#         sf.grid(row=1, column=1, sticky='w')
#         for s in (1, 2, 3):
#             tk.Radiobutton(sf, text=str(s), variable=self._arch_stage_var,
#                            value=s).pack(side='left', padx=4)
#
#         # ── constellation settings ────────────────────────────────────────────
#         lf_const = tk.LabelFrame(parent, text='Constellation settings', **pad)
#         lf_const.grid(row=1, column=0, sticky='nsew', **pad)
#
#         def _erow(lf, label, default, r):
#             tk.Label(lf, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
#             v = tk.StringVar(value=default)
#             tk.Entry(lf, textvariable=v, width=10).grid(row=r, column=1, **pad)
#             return v
#
#         self._arch_snr    = _erow(lf_const, 'SNR (dB)',              '10', 0)
#         self._arch_itr    = _erow(lf_const, 'Noise avg iterations', '1',  1)
#
#         # ── output ────────────────────────────────────────────────────────────
#         lf_out = tk.LabelFrame(parent, text='Output', **pad)
#         lf_out.grid(row=1, column=1, sticky='nsew', **pad)
#
#         tk.Label(lf_out, text='Save plot to (optional)', anchor='w').grid(
#             row=0, column=0, sticky='w', **pad)
#         self._arch_save = tk.StringVar()
#         tk.Entry(lf_out, textvariable=self._arch_save, width=28).grid(row=0, column=1, **pad)
#         tk.Button(lf_out, text='Browse…',
#                   command=lambda: self._browse_save(self._arch_save)).grid(row=0, column=2, **pad)
#         self._arch_show = tk.BooleanVar(value=True)
#         tk.Checkbutton(lf_out, text='Show plot interactively',
#                        variable=self._arch_show).grid(row=1, column=0,
#                                                       columnspan=3, sticky='w', **pad)
#
#         # ── run buttons ───────────────────────────────────────────────────────
#         btn_frame = tk.Frame(parent)
#         btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
#         btn_frame.columnconfigure(0, weight=1)
#         btn_frame.columnconfigure(1, weight=1)
#
#         tk.Button(btn_frame, text='▶  Plot Architecture', font=('', 11, 'bold'),
#                   bg='#8E44AD', fg='white',
#                   command=self._arch_run_topology).grid(row=0, column=0, sticky='ew', padx=(0, 4))
#         tk.Button(btn_frame, text='▶  Plot Constellation', font=('', 11, 'bold'),
#                   bg='#E67E22', fg='white',
#                   command=self._arch_run_constellation).grid(row=0, column=1, sticky='ew', padx=(4, 0))
#
#         parent.columnconfigure(0, weight=1)
#         parent.columnconfigure(1, weight=1)
#
#     # ── arch helpers ──────────────────────────────────────────────────────────
#
#     def _arch_browse(self):
#         path = filedialog.askdirectory(title='Select model folder')
#         if path:
#             self._arch_folder_entry.configure(state='normal')
#             self._arch_folder_var.set(path)
#             self._arch_folder_entry.configure(state='readonly')
#
#     def _arch_load(self):
#         path = self._arch_folder_var.get().strip()
#         if not path:
#             messagebox.showwarning('No folder', 'Please select a model folder.')
#             return None
#         stage = self._arch_stage_var.get()
#         try:
#             model = load_model(path, stage)
#             return model
#         except Exception as e:
#             messagebox.showerror('Load error', f'Failed to load model:\n{e}')
#             return None
#
#     def _arch_run_topology(self):
#         model = self._arch_load()
#         if model is None:
#             return
#         folder = self._arch_folder_var.get().strip()
#         save_path = self._arch_save.get().strip() or None
#         plot_architecture(model, folder_path=folder, save_path=save_path, show=self._arch_show.get())
#
#     def _arch_run_constellation(self):
#         model = self._arch_load()
#         if model is None:
#             return
#         try:
#             snr_db = float(self._arch_snr.get())
#             n_itr  = int(self._arch_itr.get())
#         except ValueError as e:
#             messagebox.showerror('Invalid setting', str(e))
#             return
#         save_path = self._arch_save.get().strip() or None
#         plot_constellation(model, snr_db=snr_db, n_itr=n_itr,
#                            save_path=save_path, show=self._arch_show.get())
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────
#
# if __name__ == '__main__':
#     app = App()
#     app.mainloop()

"""
compare_models.py
=================
Two-tab GUI tool for offline model analysis.

Tab 1 – BER Comparison
    Load trained model folders, evaluate BER vs SNR, overlay results.

Tab 2 – Log Comparison
    Load stage-2 log .pt files (stage2_all_epochs_raw.pt) from one or more
    models and overlay any logged metric on a single graph.
    Special case: when only ONE model is loaded, two log files from that
    model can be selected and plotted together (e.g. two training runs).
"""

import os
import sys
import tkinter as tk          # kept only for BooleanVar / StringVar / IntVar
from tkinter import filedialog, messagebox
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

# All keys present in the log dict that can be plotted.
# (channel-indexed ones are expanded dynamically after loading)
LOG_SCALAR_KEYS = ['loss', 'v_score', 'n_drops', 'worst_BER']
LOG_PER_CH_KEYS = ['BER_per_ch', 'n_relays_per_ch',
                   'w_norm_per_ch', 'b_norm_per_ch', 'p_entropy_per_ch']

METRIC_LABELS = {
    'loss':           'Total loss',
    'v_score':        'V score',
    'n_drops':        'Weight drops / 100 itr',
    'worst_BER':      'Worst BER (all channels)',
    'BER_per_ch':     'Worst BER',
    'n_relays_per_ch':'Active relays (P>0.9)',
    'w_norm_per_ch':  'Mean |w|',
    'b_norm_per_ch':  'Mean |b|',
    'p_entropy_per_ch':'P entropy (nats)',
}

USE_LOG_SCALE = {'worst_BER', 'BER_per_ch'}

# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def read_max_snr_stage1(model_folder: str) -> float | None:
    """
    Read max_snr_train_stage_1.pt from <model_folder>/data/.
    Returns the float value, or None if the file doesn't exist.
    """
    path = os.path.join(model_folder, 'data', 'max_snr_train_stage_1.pt')
    if not os.path.isfile(path):
        return None
    try:
        val = torch.load(path, weights_only=True)
        return float(val)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BER helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def dB2lin(x):
    if not torch.is_tensor(x):
        x = torch.tensor(float(x))
    return torch.pow(10.0, x / 10.0)


def load_model(path, stage):
    def _pt(name):
        return torch.load(os.path.join(path, 'data', name + '.pt'), weights_only=True)
    MatcgRR           = _pt('MatcgRR')
    MatcgSR           = _pt('MatcgSR')
    cgRU              = _pt('cgRU')
    connectaionMatrix = _pt('connectaionMatrix')
    N_users           = _pt('N_users')
    N_channels        = int(_pt('N_channels'))
    N_relays          = int(_pt('N_relays'))
    demod_type        = _pt('demod_type')
    N_rx             = _pt('N_rx')
    N_tx            = _pt('N_tx')
    cgTU             = _pt('cgTU')
    modCode_order     = [2 ** int(n) for n in N_users]
    model = Network_multy_channel(N_users=N_users,
                                  N_relays=N_relays,
                                  N_channels=N_channels,
                                  connectaionMatrix=connectaionMatrix,
                                  MatcgSR=MatcgSR,
                                  MatcgRR=MatcgRR,
                                  MatcgRU=cgRU,
                                  MatcgTU=cgTU,
                                  modCode_order=modCode_order,
                                  demod_type=demod_type,
                                  N_rx=N_rx,
                                  N_tx=N_tx
                                  )
    model.load(path + '\\', f'stage_{stage}')
    model.eval()
    return model


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
                    signal,bits = model.sub_networks[c].modulator(batch)
                    rm   = model.sub_networks[c](signal,bits)
                    pred = model.sub_networks[c].demodulator(rm)
                    worst, _, _ = model.sub_networks[c].BER(bits=bits, pred=pred)
                    acc += float(worst)
                out[c, si] = acc / num_itr
                print(f'  ch{c} SNR={snr:.1f} dB  BER={out[c,si]:.3e}')
                if out[c, si] == 0:
                    break
    return out


def plot_ber_comparison(results, snr_range, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title('BER vs SNR — model comparison', fontsize=14)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER (log scale)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    snr_np    = snr_range.numpy()
    color_idx = 0
    for entry in results:
        name  = entry['name']
        stage = entry['stage']
        ber   = entry['ber']
        C     = ber.shape[0]
        color  = COLORS[color_idx % len(COLORS)]
        marker = MARKERS[color_idx % len(MARKERS)]
        color_idx += 1
        if C == 1:
            ax.semilogy(snr_np, ber[0].numpy(), marker=marker,
                        color=color, linewidth=1.8, markersize=5,
                        label=f'{name}  (stage {stage})')
        else:
            worst_overall = ber.max(dim=0).values.numpy()
            ax.semilogy(snr_np, worst_overall, marker=marker,
                        color=color, linewidth=2, markersize=5,
                        label=f'{name}  (stage {stage}, worst ch)')
            for c in range(C):
                ax.semilogy(snr_np, ber[c].numpy(), linestyle='--',
                            color=color, linewidth=0.9, alpha=0.55,
                            label=f'{name}  (stage {stage}, ch{c})')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_log(filepath):
    """Load a stage2_all_epochs_raw.pt log dict."""
    data = torch.load(filepath, weights_only=False)
    # unwrap if saved inside a wrapper dict (stage2_epochXX_raw.pt format)
    if 'log' in data and isinstance(data['log'], dict):
        data = data['log']
    return data


def expand_metric_names(log):
    """
    Return a list of (display_name, key, channel_or_None) tuples
    for every plottable series in the log.
    """
    names = []
    for k in LOG_SCALAR_KEYS:
        if k in log:
            names.append((METRIC_LABELS.get(k, k), k, None))
    for k in LOG_PER_CH_KEYS:
        if k in log:
            ch_dict = log[k]
            for c in sorted(ch_dict.keys()):
                display = f'{METRIC_LABELS.get(k, k)} — ch {c}'
                names.append((display, k, c))
    return names


def get_series(log, key, channel):
    """Extract (itr_array, values_array) from a log dict."""
    itr = np.array(log['itr_axis'])
    if channel is None:
        vals = np.array(log[key], dtype=float)
    else:
        vals = np.array(log[key][channel], dtype=float)
    return itr, vals


def plot_log_comparison(entries, metric_key, metric_channel,
                        window=10, save_path=None, show=True):
    """
    entries : list of {label, log}
    metric_key / metric_channel : which series to plot
    """
    use_log  = metric_key in USE_LOG_SCALE
    ma_kern  = np.ones(window) / window
    fig, ax  = plt.subplots(figsize=(12, 5))
    title    = METRIC_LABELS.get(metric_key, metric_key)
    if metric_channel is not None:
        title += f' — ch {metric_channel}'
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    plot_fn = ax.semilogy if use_log else ax.plot
    color_idx = 0

    for entry in entries:
        label = entry['label']
        log   = entry['log']
        color  = COLORS[color_idx % len(COLORS)]
        marker = MARKERS[color_idx % len(MARKERS)]
        color_idx += 1

        itr, vals = get_series(log, metric_key, metric_channel)
        # mask zeros for log-scale plots
        if use_log:
            vals = np.where(vals == 0, np.nan, vals)

        plot_fn(itr, vals, color=color, alpha=0.3, linewidth=1)
        valid = ~np.isnan(vals)
        if valid.sum() >= window:
            ma = np.convolve(vals[valid], ma_kern, mode='valid')
            plot_fn(itr[valid][window - 1:], ma,
                    color=color, linewidth=2,
                    marker=marker, markevery=max(1, len(ma)//15),
                    markersize=5, label=label)
        else:
            plot_fn(itr[valid], vals[valid], color=color,
                    linewidth=2, marker=marker, markersize=5, label=label)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Log plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_dual_metric(entry, m1_key, m1_ch, m2_key, m2_ch,
                      window=10, save_path=None, show=True):
    """
    Plot two metrics from a single log on twin y-axes.
    Left axis  = metric 1 (blue tones)
    Right axis = metric 2 (orange tones)
    """
    log      = entry['log']
    label    = entry['label']
    ma_kern  = np.ones(window) / window

    itr = np.array(log['itr_axis'])

    def _prep(key, ch):
        _, vals = get_series(log, key, ch)
        if key in USE_LOG_SCALE:
            vals = np.where(vals == 0, np.nan, vals)
        return vals

    v1 = _prep(m1_key, m1_ch)
    v2 = _prep(m2_key, m2_ch)

    t1 = METRIC_LABELS.get(m1_key, m1_key) + (f' — ch {m1_ch}' if m1_ch is not None else '')
    t2 = METRIC_LABELS.get(m2_key, m2_key) + (f' — ch {m2_ch}' if m2_ch is not None else '')

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'{label}', fontsize=12)

    c1 = COLORS[0]   # blue family
    c2 = COLORS[1]   # orange family

    # ── left axis ─────────────────────────────────────────────────────────────
    pf1 = ax1.semilogy if m1_key in USE_LOG_SCALE else ax1.plot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(t1, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)

    pf1(itr, v1, color=c1, alpha=0.25, linewidth=1)
    valid1 = ~np.isnan(v1)
    if valid1.sum() >= window:
        ma1 = np.convolve(v1[valid1], ma_kern, mode='valid')
        pf1(itr[valid1][window - 1:], ma1, color=c1, linewidth=2,
            marker='o', markevery=max(1, len(ma1) // 15),
            markersize=5, label=t1)
    else:
        pf1(itr[valid1], v1[valid1], color=c1, linewidth=2, label=t1)

    # ── right axis ────────────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    pf2 = ax2.semilogy if m2_key in USE_LOG_SCALE else ax2.plot
    ax2.set_ylabel(t2, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)

    pf2(itr, v2, color=c2, alpha=0.25, linewidth=1)
    valid2 = ~np.isnan(v2)
    if valid2.sum() >= window:
        ma2 = np.convolve(v2[valid2], ma_kern, mode='valid')
        pf2(itr[valid2][window - 1:], ma2, color=c2, linewidth=2,
            linestyle='--', marker='^',
            markevery=max(1, len(ma2) // 15),
            markersize=5, label=t2)
    else:
        pf2(itr[valid2], v2[valid2], color=c2, linewidth=2,
            linestyle='--', label=t2)

    # combined legend outside
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left',
               bbox_to_anchor=(1.08, 1), borderaxespad=0, fontsize=9)

    ax1.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 0.80, 1])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Dual-metric plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variation helpers  (Tab 3)
# ─────────────────────────────────────────────────────────────────────────────

def discover_subfolders(main_folder: str) -> list[str]:
    """
    Return sorted list of immediate sub-folder names inside *main_folder*
    that contain at least one model with outputs/SNR.pt + outputs/worst_BER.pt.
    """
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


def discover_models_in_subfolders(main_folder: str,
                                   subfolders: list[str]) -> list[dict]:
    """
    For each sub-folder name in *subfolders*, walk its model directories and
    collect every model that has outputs/SNR.pt + outputs/worst_BER.pt.

    Returns a list of dicts:
        { 'label':        '<subfolder> / <model_name>',
          'outputs_path': full path to the outputs folder }
    """
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
                found.append({'label': f'{sf} / {model_name}',
                              'outputs_path': out})
    return found


def load_variation_ber(outputs_path: str,
                       snr_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Load SNR.pt and worst_BER.pt from *outputs_path*.
    For each value in *snr_targets* find the closest recorded SNR point and
    return matched (snr_matched, ber_matched) arrays.

    worst_BER.pt may be shape [N_snr] or [N_channels, N_snr]; the latter is
    reduced to the worst channel (max over axis 0).
    """
    snr_raw = torch.load(os.path.join(outputs_path, 'SNR.pt'), weights_only=True)
    ber_raw = torch.load(os.path.join(outputs_path, 'worst_BER.pt'), weights_only=True)

    snr_np = (snr_raw.numpy() if torch.is_tensor(snr_raw) else np.array(snr_raw)).flatten()
    ber_np = (ber_raw.numpy() if torch.is_tensor(ber_raw) else np.array(ber_raw))
    if ber_np.ndim == 2:
        ber_np = ber_np.max(axis=0)
    ber_np = ber_np.flatten()

    matched_snr, matched_ber = [], []
    for t in snr_targets:
        idx = int(np.argmin(np.abs(snr_np - t)))
        matched_snr.append(snr_np[idx])
        matched_ber.append(ber_np[idx])
    return np.array(matched_snr), np.array(matched_ber)


def plot_variation_all(results: list[dict], main_bers=None, save_path=None, show=True):
    """
    Plot every individual model BER curve, plus optionally the main model.
    results:  list of { 'label', 'snr': np.ndarray, 'ber': np.ndarray }
    main_bers: list of { 'label', 'snr': np.ndarray, 'ber': np.ndarray } or None

    Paper-ready formatting:
      • Double-column IEEE width (7 in × 4.5 in)
      • Legend inside the axes (upper right) — ncol=2 if many entries
      • Font sizes matching a 10 pt document
      • 300 dpi / vector-friendly tight layout
    To adjust: change PAPER_W/PAPER_H, FONT_* constants, or legend loc/ncol below.
    """
    # ── paper layout constants ── tweak these to taste ──────────────────────
    PAPER_W   = 7.0          # figure width  in inches (3.5 = single col, 7 = double col)
    PAPER_H   = 4.5          # figure height in inches
    FONT_AX   = 11           # axis-label font size
    FONT_TICK = 10           # tick-label font size
    FONT_LEG  = 8            # legend font size
    SAVE_DPI  = 300          # resolution for raster formats (png); irrelevant for pdf/eps
    LEG_LOC   = 'upper right'  # legend anchor inside the axes
    LEG_NCOL  = 2            # legend columns (increase to fit more entries per row)
    # ────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
    ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
    ax.set_ylabel('Worst BER', fontsize=FONT_AX)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    print(f"found {len(results)} models to compare")

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

    ylim = 1e-10
    main_styles = [
        dict(color='black',   linestyle='-',  marker='o'),
        dict(color='dimgray', linestyle='--', marker='s'),
    ]
    for j, mb in enumerate(main_bers or []):
        snr, ber = mb['snr'], mb['ber']
        ber_plot = np.where(ber > 0, ber, np.nan)
        if ~np.isnan(ber_plot).all():
            ylim = max(ylim, ber_plot[~np.isnan(ber_plot)].min())
        st = main_styles[j % len(main_styles)]
        ax.semilogy(snr, ber_plot, linewidth=2.0, markersize=5,
                    label=mb['label'], zorder=5, **st)

    ax.set_ylim([ylim, 1])
    ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
              handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
              columnspacing=1.0, framealpha=0.85)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
        print(f'All-models plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_variation_envelope(folder_results: list[dict], main_bers=None, save_path=None, show=True):
    """
    For each sub-folder plot worst/best BER envelope with filled area.
    Single-model folders draw one line (no best/worst suffix).

    folder_results: list of {
        'label':   sub-folder name,
        'snr':     np.ndarray  [N_snr],
        'ber_mat': np.ndarray  [N_models, N_snr]
    }

    Paper-ready formatting:
      • Double-column IEEE width (7 in × 4.5 in)
      • Legend inside the axes — ncol=2 keeps it compact for two-column papers
      • Font sizes matching a 10 pt document
      • 300 dpi tight layout
    To adjust: change PAPER_W/PAPER_H, FONT_* constants, or LEG_LOC/LEG_NCOL below.
    """
    # ── paper layout constants ── tweak these to taste ──────────────────────
    PAPER_W   = 7.0          # figure width  in inches (3.5 = single col, 7 = double col)
    PAPER_H   = 4.5          # figure height in inches
    FONT_AX   = 11           # axis-label font size
    FONT_TICK = 10           # tick-label font size
    FONT_LEG  = 8            # legend font size
    SAVE_DPI  = 300          # dpi for raster formats; irrelevant for pdf/eps
    LEG_LOC   = 'upper right'  # legend anchor inside the axes
    LEG_NCOL  = 2            # legend columns (increase if many entries)
    # ────────────────────────────────────────────────────────────────────────

    import warnings
    fig, ax = plt.subplots(figsize=(PAPER_W, PAPER_H))
    ax.set_xlabel('SNR (dB)', fontsize=FONT_AX)
    ax.set_ylabel('Worst BER', fontsize=FONT_AX)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.6)
    print(f"found {len(folder_results)} sub-folders to compare")

    for i, entry in enumerate(folder_results):
        color    = COLORS[i % len(COLORS)]
        snr      = entry['snr']
        mat      = entry['ber_mat']
        n_models = mat.shape[0]

        mat_safe = np.where(mat > 0, mat, np.nan)

        all_nan_cols = np.all(np.isnan(mat_safe), axis=0)
        if all_nan_cols.any():
            bad_snr = snr[all_nan_cols]
            print(f'  Warning [{entry["label"]}]: all models have BER=0 '
                  f'at SNR = {bad_snr} dB — those points will be skipped.')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    message='All-NaN slice encountered')
            worst = np.nanmax(mat_safe, axis=0)
            best  = np.nanmin(mat_safe, axis=0)

        valid = ~(np.isnan(worst) | np.isnan(best))

        if n_models == 1:
            ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
                        marker='o', markersize=4, markevery=1,
                        label=entry['label'])
        else:
            ax.semilogy(snr[valid], worst[valid], color=color, linewidth=1.5,
                        marker='^', markersize=4, markevery=1,
                        label=f'{entry["label"]}  worst')
            ax.semilogy(snr[valid], best[valid], color=color, linewidth=1.5,
                        linestyle='--', marker='v', markersize=4, markevery=1,
                        label=f'{entry["label"]}  best')
            if valid.any():
                ax.fill_between(snr[valid], best[valid], worst[valid],
                                color=color, alpha=0.18)

    ylim = 1e-10
    main_styles = [
        dict(color='black',   linestyle='-',  marker='o'),
        dict(color='dimgray', linestyle='--', marker='s'),
    ]
    for j, mb in enumerate(main_bers or []):
        snr, ber = mb['snr'], mb['ber']
        ber_plot = np.where(ber > 0, ber, np.nan)
        if ~np.isnan(ber_plot).all():
            ylim = max(ylim, ber_plot[~np.isnan(ber_plot)].min())
        st = main_styles[j % len(main_styles)]
        ax.semilogy(snr, ber_plot, linewidth=2.0, markersize=5,
                    label=mb['label'], zorder=5, **st)

    ax.set_ylim([ylim, 1])
    ax.legend(loc=LEG_LOC, ncol=LEG_NCOL, fontsize=FONT_LEG,
              handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
              columnspacing=1.0, framealpha=0.85)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=SAVE_DPI)
        print(f'Envelope plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
# Architecture & Constellation helpers  (Tab 4)
# ─────────────────────────────────────────────────────────────────────────────

def int_to_binary(integers, n_bits):
    """Convert integer tensor to binary matrix [N, n_bits] MSB-first."""
    mask = 2 ** torch.arange(n_bits - 1, -1, -1)
    return ((integers.unsqueeze(-1) & mask) > 0).int()


def plot_architecture(model, folder_path, save_path=None, show=True):
    """
    Spatial scatter plot.
    TX  — upward triangle  (^)  large, with black edge
    RX  — downward triangle (v)  large, with black edge
    Relay — circle (o)
    Each channel gets a distinct jet color. Relays colored by dominant channel (P >= 0.5).
    """
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
        col = colors(c)

        plt.scatter(float(posT[c, 0]), float(posT[c, 1]),
                    marker='^', label=f'TX - channel {c}',
                    color=col, s=300, zorder=5,
                    edgecolors='black', linewidths=0.6)

        mask = np.squeeze(color_RR == c + 1)
        if mask.any():
            plt.scatter(posR[mask, 0].numpy(), posR[mask, 1].numpy(),
                        marker='o', color=col,
                        label=f'relay - channel - {c}', s=200, zorder=3,
                        edgecolors='black', linewidths=0.4)

        pu = posU[c]
        pu_np = pu.numpy() if torch.is_tensor(pu) else np.array(pu)
        if pu_np.ndim == 1:
            pu_np = pu_np.reshape(1, -1)
        plt.scatter(pu_np[:, 0], pu_np[:, 1],
                    marker='v', color=col,
                    label=f'RX - channel - {c}', s=300, zorder=5,
                    edgecolors='black', linewidths=0.6)

    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncols=3 * C, mode='expand', borderaxespad=0.)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Architecture plot saved to {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_constellation(model, snr_db=10.0, n_itr=1, save_path=None, show=True):
    """
    One figure per channel, one figure per RX user within that channel.

    Figure 1 — TX antennas (one subplot per antenna, side by side):
        Shows the transmitted constellation with bit-string labels.

    Figure 2..N_users+1 — per RX user:
        Grid of N_symbols subplots (one per transmitted symbol).
        Each cell shows the received cloud for that specific symbol only,
        so overlapping symbols are split apart into separate panels.
        Color inside each panel = transmit symbol color (consistent with TX figure).

    n_itr > 1 stacks multiple noisy passes to show the full noise cloud.
    """
    import matplotlib.patches as mpatches
    import math as _math

    snr_lin  = dB2lin(snr_db)
    C        = model.N_channels
    sym_cmap = plt.get_cmap('tab20')

    for c in range(C):
        sub       = model.sub_networks[c]
        sub.SNR   = snr_lin
        N_users_c = int(model.N_users[c])
        N_symbols = 2 ** N_users_c

        # ── build symbol set ──────────────────────────────────────────────────
        bits = int_to_binary(torch.arange(0, N_symbols), N_users_c).to(torch.complex64)
        if model.demod_type == 'complex':
            s = sub.transmitNN(bits).detach()
        else:
            symbols_int = torch.sum(
                2 ** torch.unsqueeze(
                    torch.linspace(0, N_users_c - 1, N_users_c), dim=1)
                * bits.T, dim=0
            ).real.to(torch.int32)
            s = sub.modulation[symbols_int]   # [N_symbols] complex

        print(f'Channel {c}: max TX amplitude^2 = {torch.max(s.abs() ** 2).item():.4f}')

        s_plot = s.unsqueeze(1) if model.demod_type == 'simple' else s
        if s_plot.ndim == 1:
            s_plot = s_plot.unsqueeze(1)
        N_tx_plot = s_plot.shape[1]

        # ── forward passes ────────────────────────────────────────────────────
        rm_runs = []
        with torch.no_grad():
            for _ in range(n_itr):
                if model.demod_type == 'complex':
                    rm = sub(s, bits.real.T).detach().T  # [N_symbols, N_users_c]
                else:
                    rm = torch.squeeze(sub(s, bits.real.T).detach(), 1).T

                rm_runs.append(rm)

        bits_labels = [
            ''.join(str(b) for b in
                    int_to_binary(torch.tensor([sym]), N_users_c)[0].int().tolist())
            for sym in range(N_symbols)
        ]
        sym_colors = [sym_cmap(sym / max(N_symbols - 1, 1)) for sym in range(N_symbols)]

        def _save(fig, suffix):
            if save_path:
                base, ext = os.path.splitext(save_path)
                ext = ext or '.png'
                p = f'{base}_ch{c}_{suffix}{ext}'
                fig.savefig(p, bbox_inches='tight', dpi=150)
                print(f'Saved: {p}')
                plt.close(fig)

        # ── Figure 1: TX constellation ────────────────────────────────────────
        fig_tx, axes_tx = plt.subplots(1, N_tx_plot,
                                       figsize=(6 * N_tx_plot, 6),
                                       squeeze=False)
        fig_tx.suptitle(f'Channel {c}  |  TX  |  SNR = {snr_db:.1f} dB',
                        fontsize=13)

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
            ax.set_xlabel('Real');  ax.set_ylabel('Imag')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
                       for s in range(N_symbols)]
            ax.legend(handles=handles, title='Symbol', fontsize=7,
                      ncol=max(1, N_symbols // 4), loc='best')

        fig_tx.tight_layout()
        _save(fig_tx, 'TX')

        # ── One figure per RX user — all symbols on a single plot ──────────
        for u in range(N_users_c):
            fig_rx, ax = plt.subplots(figsize=(7, 7))
            fig_rx.suptitle(
                f'Channel {c}  |  RX user {u}  |  SNR = {snr_db:.1f} dB  '
                f'({n_itr} iteration{"s" if n_itr > 1 else ""})',
                fontsize=13)

            for sym in range(N_symbols):
                pts = torch.stack([rm_runs[it][sym, u] for it in range(n_itr)])
                ax.scatter(pts.real.numpy(), pts.imag.numpy(),
                           color=sym_colors[sym], s=20, alpha=0.6,
                           zorder=3, label=bits_labels[sym])
                # centroid marker
                cx = pts.real.mean().item()
                cy = pts.imag.mean().item()
                ax.scatter([cx], [cy], color=sym_colors[sym], s=80,
                           marker='+', zorder=5, linewidths=2)
                ax.annotate(bits_labels[sym], (cx, cy),
                            textcoords='offset points', xytext=(6, 6),
                            fontsize=9, fontweight='bold',
                            color=sym_colors[sym])

            ax.set_xlabel('Real', fontsize=10)
            ax.set_ylabel('Imag', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            handles = [mpatches.Patch(color=sym_colors[s], label=bits_labels[s])
                       for s in range(N_symbols)]
            ax.legend(handles=handles, title='Symbol', fontsize=8,
                      ncol=max(1, N_symbols // 4), loc='best')

            fig_rx.tight_layout()
            _save(fig_rx, f'RX_user{u}')
    if show:
        plt.show()




# GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Model Comparison Tool')
        self.resizable(True, True)
        self._ber_rows: list[dict] = []
        self._log_rows: list[dict] = []
        self._build_ui()

    # ── top-level notebook ────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ctk.CTkTabview(self)
        nb.pack(fill='both', expand=True, padx=6, pady=6)

        nb.add('  BER Comparison  ')
        nb.add('  Log Comparison  ')
        nb.add('  Variation Comparison  ')
        nb.add('  Architecture & Constellation  ')

        self._tab_ber  = nb.tab('  BER Comparison  ')
        self._tab_log  = nb.tab('  Log Comparison  ')
        self._tab_var  = nb.tab('  Variation Comparison  ')
        self._tab_arch = nb.tab('  Architecture & Constellation  ')

        self._build_ber_tab(self._tab_ber)
        self._build_log_tab(self._tab_log)
        self._build_var_tab(self._tab_var)
        self._build_arch_tab(self._tab_arch)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — BER
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ber_tab(self, parent):
        pad = dict(padx=8, pady=4)

        lf_models = ctk.CTkFrame(parent)
        lf_models.grid(row=0, column=0, columnspan=2, sticky='nsew', **pad)
        ctk.CTkLabel(lf_models, text='Models', font=ctk.CTkFont(size=13, weight='bold')).pack(anchor='w', padx=8, pady=(6,2))
        self._ber_inner = ctk.CTkScrollableFrame(lf_models, height=120)
        self._ber_inner.pack(fill='both', expand=True, padx=4, pady=4)
        ctk.CTkButton(lf_models, text='＋  Add model folder',
                      command=self._ber_add).pack(anchor='w', **pad)

        lf_eval = ctk.CTkFrame(parent)
        lf_eval.grid(row=1, column=0, sticky='nsew', **pad)
        ctk.CTkLabel(lf_eval, text='Evaluation settings', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(6,2))

        def _erow(label, default, r):
            ctk.CTkLabel(lf_eval, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
            v = tk.StringVar(value=default)
            ctk.CTkEntry(lf_eval, textvariable=v, width=90).grid(row=r, column=1, **pad)
            return v

        self._snr_min  = _erow('SNR min (dB)',   '-20',  1)
        self._snr_max  = _erow('SNR max (dB)',   '40',   2)
        self._snr_step = _erow('SNR step (dB)',  '1',    3)
        self._batch    = _erow('Batch size',     '1000', 4)
        self._n_itr    = _erow('Avg iterations', '10',   5)

        lf_out = ctk.CTkFrame(parent)
        lf_out.grid(row=1, column=1, sticky='nsew', **pad)
        ctk.CTkLabel(lf_out, text='Output', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2))
        ctk.CTkLabel(lf_out, text='Save plot to (optional)', anchor='w').grid(
            row=1, column=0, sticky='w', **pad)
        self._ber_save = tk.StringVar()
        ctk.CTkEntry(lf_out, textvariable=self._ber_save, width=220).grid(row=1, column=1, **pad)
        ctk.CTkButton(lf_out, text='Browse…', width=80,
                      command=lambda: self._browse_save(self._ber_save)).grid(row=1, column=2, **pad)
        self._ber_show = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(lf_out, text='Show plot interactively',
                        variable=self._ber_show, onvalue=True, offvalue=False).grid(
            row=2, column=0, columnspan=3, sticky='w', **pad)

        ctk.CTkButton(parent, text='▶  Run BER comparison',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#2E86AB', hover_color='#1a5f7a',
                      command=self._ber_run).grid(row=2, column=0, columnspan=2,
                                                  sticky='ew', padx=8, pady=8)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def _ber_add(self, path=''):
        if not path:
            path = filedialog.askdirectory(title='Select model folder')
        if not path:
            return
        frame = ctk.CTkFrame(self._ber_inner, fg_color='transparent')
        frame.pack(fill='x', pady=2)
        display = path if len(path) <= 55 else '…' + path[-52:]
        ctk.CTkLabel(frame, text=display, anchor='w', width=380,
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky='w')

        ctk.CTkLabel(frame, text='Stages:').grid(row=0, column=1, padx=(8, 2))
        stage_vars = {}
        for col, s in enumerate((1, 2, 3), start=2):
            var = tk.BooleanVar(value=(s == 3))
            ctk.CTkCheckBox(frame, text=str(s), variable=var,
                            onvalue=True, offvalue=False, width=50).grid(row=0, column=col, padx=2)
            stage_vars[s] = var

        ctk.CTkLabel(frame, text='Label:').grid(row=0, column=5, padx=(8, 2))
        label_var = tk.StringVar(value=os.path.basename(path.rstrip('/\\')))
        ctk.CTkEntry(frame, textvariable=label_var, width=200).grid(row=0, column=6, padx=2)

        def _remove(f=frame):
            f.destroy()
            self._ber_rows = [r for r in self._ber_rows if r['frame'] is not f]

        ctk.CTkButton(frame, text='✕', width=30, fg_color='#c0392b', hover_color='#922b21',
                      command=_remove).grid(row=0, column=7, padx=(8, 0))
        self._ber_rows.append(dict(path=path, stage_vars=stage_vars,
                                   label_var=label_var, frame=frame))
        if len(self._ber_rows) == 1:
            max_snr = read_max_snr_stage1(path)
            if max_snr is not None:
                self._snr_min.set(str(int(round(max_snr - 10))))

    def _ber_run(self):
        rows = [r for r in self._ber_rows if r['frame'].winfo_exists()]
        if not rows:
            messagebox.showwarning('No models', 'Add at least one model folder.')
            return
        try:
            snr_min  = float(self._snr_min.get())
            snr_max  = float(self._snr_max.get())
            snr_step = float(self._snr_step.get())
            batch    = int(self._batch.get())
            num_itr  = int(self._n_itr.get())
            snr_range = torch.arange(snr_min, snr_max + snr_step * 0.5, snr_step)
        except ValueError as e:
            messagebox.showerror('Invalid setting', str(e)); return

        save_path = self._ber_save.get().strip() or None
        results   = []
        for row in rows:
            path  = row['path']
            name  = row['label_var'].get().strip() or os.path.basename(path.rstrip('/\\'))
            selected_stages = [s for s, var in row['stage_vars'].items() if var.get()]
            if not selected_stages:
                messagebox.showwarning('No stage selected',
                                       f'Select at least one stage for:\n{name}')
                continue
            for stage in selected_stages:
                print(f'\n{"="*55}\n  Loading: {name}  |  stage {stage}\n{"="*55}')
                try:
                    model = load_model(path, stage)
                except Exception as e:
                    messagebox.showerror('Load error',
                                         f'Failed to load stage {stage}:\n{path}\n\n{e}')
                    continue
                ber = evaluate_model(model, snr_range, batch, num_itr)
                results.append(dict(name=name, stage=stage, ber=ber))
                if save_path:
                    base = os.path.splitext(save_path)[0]
                    safe = name.replace(' ', '_').replace('\\', '_').replace('/', '_')
                    torch.save(ber, f'{base}_{safe}_stage{stage}_BER.pt')
        if not results:
            messagebox.showinfo('Done', 'No models evaluated.'); return
        plot_ber_comparison(results, snr_range,
                            save_path=save_path, show=self._ber_show.get())

    def _build_log_tab(self, parent):
        pad = dict(padx=8, pady=4)

        # ── top: log file list ────────────────────────────────────────────────
        lf_logs = ctk.CTkFrame(parent)
        lf_logs.grid(row=0, column=0, columnspan=3, sticky='nsew', **pad)
        ctk.CTkLabel(lf_logs, text='Log files', font=ctk.CTkFont(size=12, weight='bold')).pack(anchor='w', padx=8, pady=(6,2))

        self._log_inner = ctk.CTkScrollableFrame(lf_logs, height=120)
        self._log_inner.pack(fill='both', expand=True, padx=4, pady=4)

        btn_row = ctk.CTkFrame(lf_logs, fg_color='transparent')
        btn_row.pack(fill='x', pady=(4, 0))
        ctk.CTkButton(btn_row, text='＋  Add log file (.pt)',
                      command=self._log_add_file).pack(side='left', **pad)
        ctk.CTkButton(btn_row, text='＋  Add model folder  (auto-finds log)',
                      command=self._log_add_folder).pack(side='left', **pad)

        # ── middle: metric selectors ──────────────────────────────────────────
        lf_metric = ctk.CTkFrame(parent)
        lf_metric.grid(row=1, column=0, sticky='nsew', **pad)
        ctk.CTkLabel(lf_metric, text='Metrics to plot', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2))

        ctk.CTkLabel(lf_metric, text='Left axis (Y1):', anchor='w',
                     font=ctk.CTkFont(weight='bold')).grid(row=1, column=0, sticky='w', **pad)
        self._metric_var = tk.StringVar()
        self._metric_cb  = ctk.CTkComboBox(lf_metric, variable=self._metric_var,
                                            state='readonly', width=280)
        self._metric_cb.grid(row=1, column=1, sticky='w', **pad)

        self._metric2_label = ctk.CTkLabel(lf_metric, text='Right axis (Y2):', anchor='w',
                                            font=ctk.CTkFont(weight='bold'))
        self._metric2_label.grid(row=2, column=0, sticky='w', **pad)
        self._metric2_var = tk.StringVar()
        self._metric2_cb  = ctk.CTkComboBox(lf_metric, variable=self._metric2_var,
                                             state='readonly', width=280)
        self._metric2_cb.grid(row=2, column=1, sticky='w', **pad)
        self._metric2_none_lbl = ctk.CTkLabel(lf_metric,
                                               text='(available when only one log is loaded)',
                                               text_color='gray', font=ctk.CTkFont(size=10))
        self._metric2_none_lbl.grid(row=2, column=2, sticky='w', padx=4)

        ctk.CTkButton(lf_metric, text='↺  Refresh', width=90,
                      command=self._log_refresh_metrics).grid(row=1, column=2, **pad)

        ctk.CTkLabel(lf_metric, text='MA window:').grid(row=3, column=0, sticky='w', **pad)
        self._ma_var = tk.StringVar(value='10')
        ctk.CTkEntry(lf_metric, textvariable=self._ma_var, width=60).grid(row=3, column=1,
                                                                            sticky='w', **pad)

        # ── right: output ─────────────────────────────────────────────────────
        lf_out = ctk.CTkFrame(parent)
        lf_out.grid(row=1, column=1, sticky='nsew', **pad)
        ctk.CTkLabel(lf_out, text='Output', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2))
        ctk.CTkLabel(lf_out, text='Save plot to (optional)', anchor='w').grid(
            row=1, column=0, sticky='w', **pad)
        self._log_save = tk.StringVar()
        ctk.CTkEntry(lf_out, textvariable=self._log_save, width=220).grid(row=1, column=1, **pad)
        ctk.CTkButton(lf_out, text='Browse…', width=80,
                      command=lambda: self._browse_save(self._log_save)).grid(row=1, column=2, **pad)
        self._log_show = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(lf_out, text='Show plot interactively',
                        variable=self._log_show, onvalue=True, offvalue=False).grid(
            row=2, column=0, columnspan=3, sticky='w', **pad)

        # ── run button ────────────────────────────────────────────────────────
        ctk.CTkButton(parent, text='▶  Plot logs',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#3D8A40', hover_color='#27632a',
                      command=self._log_run).grid(row=2, column=0, columnspan=3,
                                                  sticky='ew', padx=8, pady=8)

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    # ── log row management ────────────────────────────────────────────────────

    def _log_add_file(self, filepath='', label_default=''):
        """Add a row for a .pt log file."""
        if not filepath:
            filepath = filedialog.askopenfilename(
                title='Select log .pt file',
                filetypes=[('PyTorch files', '*.pt'), ('All files', '*.*')])
        if not filepath:
            return
        try:
            test = torch.load(filepath, weights_only=False)
            if 'log' in test:
                test = test['log']
            assert 'itr_axis' in test, 'No itr_axis key — not a valid log file'
        except Exception as e:
            messagebox.showerror('Invalid log file', str(e)); return

        frame = ctk.CTkFrame(self._log_inner, fg_color='transparent')
        frame.pack(fill='x', pady=2)

        display = filepath if len(filepath) <= 60 else '…' + filepath[-57:]
        ctk.CTkLabel(frame, text=display, anchor='w', width=460,
                     font=ctk.CTkFont(size=10)).grid(row=0, column=0, sticky='w')
        ctk.CTkLabel(frame, text='Label:').grid(row=0, column=1, padx=(8, 2))
        default_label = label_default or os.path.basename(
            os.path.dirname(filepath)).replace('_', ' ')
        label_var = tk.StringVar(value=default_label)
        ctk.CTkEntry(frame, textvariable=label_var, width=160).grid(row=0, column=2, padx=2)

        self._log_rows.append(dict(filepath=filepath, label_var=label_var, frame=frame))

        def _remove(f=frame):
            f.destroy()
            self._log_rows = [r for r in self._log_rows if r['frame'] is not f]
            self._log_refresh_metrics()

        ctk.CTkButton(frame, text='✕', width=30, fg_color='#c0392b', hover_color='#922b21',
                      command=_remove).grid(row=0, column=3, padx=(8, 0))

        self._log_refresh_metrics()

    def _log_add_folder(self):
        """Auto-find stage2_all_epochs_raw.pt inside a model folder."""
        folder = filedialog.askdirectory(title='Select model folder')
        if not folder:
            return
        candidates = [
            os.path.join(folder, 'data', 'stage2_all_epochs_raw.pt'),
            os.path.join(folder, 'stage2_all_epochs_raw.pt'),
        ]
        found = next((p for p in candidates if os.path.exists(p)), None)
        if found is None:
            found = filedialog.askopenfilename(
                initialdir=os.path.join(folder, 'data'),
                title=f'Cannot auto-find log in {folder} — select manually',
                filetypes=[('PyTorch files', '*.pt'), ('All files', '*.*')])
        if found:
            self._log_add_file(filepath=found,
                               label_default=os.path.basename(folder.rstrip('/\\')))

    def _log_refresh_metrics(self):
        """Refresh both metric dropdowns from the first loaded log."""
        valid = [r for r in self._log_rows if r['frame'].winfo_exists()]

        if not valid:
            self._metric_cb.configure(values=[])
            self._metric2_cb.configure(values=[])
            return

        try:
            log  = load_log(valid[0]['filepath'])
            opts = expand_metric_names(log)
            self._metric_map = {display: (k, ch) for display, k, ch in opts}
            names = list(self._metric_map.keys())

            self._metric_cb.configure(values=names)
            self._metric2_cb.configure(values=['(none)'] + names)

            if self._metric_var.get() not in self._metric_map:
                self._metric_cb.set(names[0] if names else '')
            if self._metric2_var.get() not in self._metric_map:
                self._metric2_cb.set('(none)')
        except Exception as e:
            messagebox.showerror('Metric refresh error', str(e))
            return

        # Enable / disable second metric depending on number of logs
        single = (len(valid) == 1)
        state  = 'readonly' if single else 'disabled'
        self._metric2_cb.configure(state=state)
        self._metric2_none_lbl.configure(
            text='' if single else '(available when only one log is loaded)')

    # ── run log plot ──────────────────────────────────────────────────────────

    def _log_run(self):
        valid = [r for r in self._log_rows if r['frame'].winfo_exists()]
        if not valid:
            messagebox.showwarning('No logs', 'Add at least one log file.'); return

        if not self._metric_var.get():
            messagebox.showwarning('No metric', 'Select a metric to plot.'); return

        if not hasattr(self, '_metric_map'):
            self._log_refresh_metrics()

        m1_display = self._metric_var.get()
        m2_display = self._metric2_var.get()
        if m1_display not in self._metric_map:
            messagebox.showerror('Error', f'Unknown metric: {m1_display}'); return

        m1_key, m1_ch = self._metric_map[m1_display]
        m2_key, m2_ch = None, None
        use_twin = (len(valid) == 1
                    and m2_display
                    and m2_display != '(none)'
                    and m2_display in self._metric_map)
        if use_twin:
            m2_key, m2_ch = self._metric_map[m2_display]

        try:
            window = int(self._ma_var.get())
        except ValueError:
            window = 10

        # ── load all logs ─────────────────────────────────────────────────────
        entries = []
        for row in valid:
            label = row['label_var'].get().strip()
            try:
                log = load_log(row['filepath'])
            except Exception as e:
                messagebox.showerror('Load error', f'{label}:\n{e}'); continue
            entries.append(dict(label=label, log=log))

        if not entries:
            return

        save_path = self._log_save.get().strip() or None

        if use_twin:
            # single log, two metrics, twin y-axes
            _plot_dual_metric(entries[0], m1_key, m1_ch, m2_key, m2_ch,
                              window=window, save_path=save_path,
                              show=self._log_show.get())
        else:
            # multiple logs, one metric
            plot_log_comparison(entries, m1_key, m1_ch,
                                window=window, save_path=save_path,
                                show=self._log_show.get())

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Variation Comparison
    # ══════════════════════════════════════════════════════════════════════════

    def _build_var_tab(self, parent):
        pad = dict(padx=8, pady=4)

        # ── main folder selector ──────────────────────────────────────────────
        lf_folder = ctk.CTkFrame(parent)
        lf_folder.grid(row=0, column=0, columnspan=2, sticky='ew', **pad)
        ctk.CTkLabel(lf_folder, text='Main model folder', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=4, padx=8, pady=(6,2), sticky='w')
        lf_folder.columnconfigure(1, weight=1)

        ctk.CTkLabel(lf_folder, text='Folder:').grid(row=1, column=0, sticky='w', **pad)
        self._var_folder_var = tk.StringVar()
        self._var_folder_entry = ctk.CTkEntry(lf_folder, textvariable=self._var_folder_var,
                                              width=400, state='readonly')
        self._var_folder_entry.grid(row=1, column=1, sticky='ew', **pad)
        ctk.CTkButton(lf_folder, text='Browse…', width=80,
                      command=self._var_browse).grid(row=1, column=2, **pad)
        ctk.CTkButton(lf_folder, text='⟳  Scan', width=80,
                      command=self._var_scan).grid(row=1, column=3, **pad)

        ctk.CTkLabel(lf_folder, text='Main model label:').grid(row=2, column=0, sticky='w', **pad)
        self._var_main_label = tk.StringVar(value='main model')
        ctk.CTkEntry(lf_folder, textvariable=self._var_main_label,
                     width=300).grid(row=2, column=1, sticky='w', **pad)

        # ── sub-folder checklist ──────────────────────────────────────────────
        lf_sf = ctk.CTkFrame(parent)
        lf_sf.grid(row=1, column=0, columnspan=2, sticky='nsew', **pad)
        ctk.CTkLabel(lf_sf, text='Sub-folders  (select which to include)',
                     font=ctk.CTkFont(size=12, weight='bold')).pack(anchor='w', padx=8, pady=(6,2))

        self._sf_inner = ctk.CTkScrollableFrame(lf_sf, height=160)
        self._sf_inner.pack(fill='both', expand=True, padx=4, pady=4)
        self._sf_entries: list[dict] = []

        sf_btn_row = ctk.CTkFrame(lf_sf, fg_color='transparent')
        sf_btn_row.pack(fill='x', pady=(2, 0))
        ctk.CTkButton(sf_btn_row, text='Select all', width=100,
                      command=self._var_sf_select_all).pack(side='left', padx=4)
        ctk.CTkButton(sf_btn_row, text='Select none', width=100,
                      command=self._var_sf_select_none).pack(side='left', padx=4)

        # ── SNR range ─────────────────────────────────────────────────────────
        lf_snr = ctk.CTkFrame(parent)
        lf_snr.grid(row=2, column=0, sticky='nsew', **pad)
        ctk.CTkLabel(lf_snr, text='SNR range  (integer steps of 1)',
                     font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(6,2))

        def _srow(label, default, r):
            ctk.CTkLabel(lf_snr, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
            v = tk.StringVar(value=default)
            ctk.CTkEntry(lf_snr, textvariable=v, width=80).grid(row=r, column=1, **pad)
            return v

        self._var_snr_min  = _srow('SNR min (dB)',  '-20', 1)
        self._var_snr_max  = _srow('SNR max (dB)',  '40',  2)
        self._var_snr_step = _srow('SNR step (dB)', '1',   3)

        # ── output ────────────────────────────────────────────────────────────
        lf_out = ctk.CTkFrame(parent)
        lf_out.grid(row=2, column=1, sticky='nsew', **pad)
        ctk.CTkLabel(lf_out, text='Output', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2))
        ctk.CTkLabel(lf_out, text='Save plot to (optional)', anchor='w').grid(
            row=1, column=0, sticky='w', **pad)
        self._var_save = tk.StringVar()
        ctk.CTkEntry(lf_out, textvariable=self._var_save, width=220).grid(row=1, column=1, **pad)
        ctk.CTkButton(lf_out, text='Browse…', width=80,
                      command=lambda: self._browse_save(self._var_save)).grid(row=1, column=2, **pad)
        self._var_show = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(lf_out, text='Show plot interactively',
                        variable=self._var_show, onvalue=True, offvalue=False).grid(
            row=2, column=0, columnspan=3, sticky='w', **pad)

        # ── run buttons ───────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(parent, fg_color='transparent')
        btn_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ctk.CTkButton(btn_frame, text='▶  Plot all models',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#27AE60', hover_color='#1e8449',
                      command=self._var_run_all).grid(row=0, column=0, sticky='ew', padx=(0, 4))
        ctk.CTkButton(btn_frame, text='▶  Plot BER envelope',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#2E86AB', hover_color='#1a5f7a',
                      command=self._var_run_envelope).grid(row=0, column=1, sticky='ew', padx=(4, 0))

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

    # ── variation helpers ─────────────────────────────────────────────────────

    def _var_browse(self):
        path = filedialog.askdirectory(title='Select main model folder')
        if path:
            self._var_folder_entry.configure(state='normal')
            self._var_folder_var.set(path)
            self._var_folder_entry.configure(state='readonly')
            self._var_scan()

    def _var_scan(self):
        """Scan main folder and populate the sub-folder checklist."""
        folder = self._var_folder_var.get().strip()
        if not folder:
            messagebox.showwarning('No folder', 'Please select a main model folder first.')
            return
        # auto-fill the main model label with the folder name (user can override)
        self._var_main_label.set(os.path.basename(folder.rstrip('/\\')))

        for w in self._sf_inner.winfo_children():
            w.destroy()
        self._sf_entries.clear()

        # auto-set SNR min from the main model's data folder (always, even if no subfolders)
        max_snr = read_max_snr_stage1(folder)
        if max_snr is not None:
            self._var_snr_min.set(str(int(round(max_snr - 10))))
            print(f'  Auto SNR min set to {int(round(max_snr - 10))} dB '
                  f'(max_snr_train_stage_1={max_snr:.1f} - 10)')

        subfolders = discover_subfolders(folder)
        if not subfolders:
            ctk.CTkLabel(self._sf_inner,
                         text='No sub-folders with valid models found.\n'
                              'Expected: <main>/<sub_folder>/<model>/outputs/{SNR.pt, worst_BER.pt}',
                         text_color='gray', justify='left').pack(anchor='w', padx=8, pady=8)
            return

        for sf_name in subfolders:
            frame = ctk.CTkFrame(self._sf_inner, fg_color='transparent')
            frame.pack(fill='x', pady=1, padx=4)
            enabled_var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(frame, text='', variable=enabled_var,
                            onvalue=True, offvalue=False, width=30).grid(row=0, column=0)
            display_var = tk.StringVar(value=sf_name)
            ctk.CTkEntry(frame, textvariable=display_var, width=300).grid(row=0, column=1, padx=4, sticky='w')
            ctk.CTkLabel(frame, text=sf_name, text_color='gray',
                         font=ctk.CTkFont(size=10)).grid(row=0, column=2, padx=(2, 8), sticky='w')
            self._sf_entries.append({'name': sf_name, 'display_var': display_var,
                                     'enabled_var': enabled_var, 'frame': frame})

        print(f'Found {len(subfolders)} sub-folder(s).')

    def _var_sf_select_all(self):
        for e in self._sf_entries:
            e['enabled_var'].set(True)

    def _var_sf_select_none(self):
        for e in self._sf_entries:
            e['enabled_var'].set(False)

    def _var_get_snr_and_active_sf(self):
        """Shared validation: returns (snr_targets, active_sf_names) or (None, None)."""
        try:
            snr_min  = float(self._var_snr_min.get())
            snr_max  = float(self._var_snr_max.get())
            snr_step = float(self._var_snr_step.get())
        except ValueError as e:
            messagebox.showerror('Invalid SNR', str(e))
            return None, None

        if snr_min >= snr_max:
            messagebox.showerror('Invalid SNR', 'SNR min must be less than SNR max.')
            return None, None
        if snr_step <= 0:
            messagebox.showerror('Invalid SNR', 'SNR step must be positive.')
            return None, None

        active_sf = [(e['name'], e['display_var'].get().strip() or e['name'])
                     for e in self._sf_entries
                     if e['frame'].winfo_exists() and e['enabled_var'].get()]
        if not active_sf:
            messagebox.showwarning('Nothing selected', 'Please select at least one sub-folder.')
            return None, None

        return np.arange(snr_min, snr_max + snr_step * 0.5, snr_step), active_sf

    def _var_load_main_ber(self, snr_targets):
        """
        Load stage-3 and stage-1 worst BER from <main_folder>/outputs/.
        Files expected:
            SNR.pt
            worst_BER_stage_3.pt   — shape [N_snr] or [N_channels, N_snr]
            worst_BER_stage_1.pt   — same shape (optional)
        Returns a list of dicts (one per stage found), each:
            {'label', 'snr': np.ndarray, 'ber': np.ndarray}
        Returns empty list if the outputs folder / SNR file is missing.
        """
        folder = self._var_folder_var.get().strip()
        out    = os.path.join(folder, 'outputs')
        snr_f  = os.path.join(out, 'SNR.pt')
        base_label = self._var_main_label.get().strip() or os.path.basename(folder.rstrip('/\\'))

        print(f'  Looking for main model outputs in: {out}')

        if not os.path.isfile(snr_f):
            messagebox.showinfo('Main model',
                                f'SNR.pt not found at:\n{snr_f}\n\n'
                                'Main model will not be plotted.')
            return []

        try:
            snr_raw = torch.load(snr_f, weights_only=True)
            snr_np  = (snr_raw.numpy() if torch.is_tensor(snr_raw)
                       else np.array(snr_raw)).flatten()
        except Exception as ex:
            messagebox.showwarning('Main model SNR load error', str(ex))
            return []

        def _load_stage_ber(stage):
            ber_f = os.path.join(out, f'worst_BER_stage_{stage}.pt')
            print(f'    worst_BER_stage_{stage}.pt exists: {os.path.isfile(ber_f)}')
            if not os.path.isfile(ber_f):
                return None
            try:
                ber_raw = torch.load(ber_f, weights_only=True)
                ber_np  = (ber_raw.numpy() if torch.is_tensor(ber_raw)
                           else np.array(ber_raw))
                if ber_np.ndim == 2:
                    ber_np = ber_np.max(axis=0)
                ber_np = ber_np.flatten()

                matched_snr, matched_ber = [], []
                for t in snr_targets:
                    idx = int(np.argmin(np.abs(snr_np - t)))
                    matched_snr.append(snr_np[idx])
                    matched_ber.append(ber_np[idx])
                snr_m = np.array(matched_snr)
                ber_m = np.array(matched_ber)
                print(f'  Main stage {stage} loaded. BER range: {ber_m.min():.2e}–{ber_m.max():.2e}')
                return {'label': f'{base_label}  (main stage {stage})',
                        'snr': snr_m, 'ber': ber_m}
            except Exception as ex:
                print(f'  Warning: could not load stage {stage} BER: {ex}')
                return None

        results = []
        for stage in (3, 1):
            entry = _load_stage_ber(stage)
            if entry is not None:
                results.append(entry)

        if not results:
            messagebox.showinfo('Main model',
                                f'No worst_BER_stage_*.pt files found in:\n{out}\n\n'
                                'Main model will not be plotted.')
        return results

    def _var_run_all(self):
        """Plot every individual model curve."""
        snr_targets, active_sf = self._var_get_snr_and_active_sf()
        if snr_targets is None:
            return

        folder   = self._var_folder_var.get().strip()
        sf_names = [name for name, _ in active_sf]
        sf_labels = {name: lbl for name, lbl in active_sf}
        models = discover_models_in_subfolders(folder, sf_names)
        if not models:
            messagebox.showinfo('No models', 'No valid models found in selected sub-folders.')
            return

        results, errors = [], []
        for m in models:
            # replace the sf_name part of the label with the user's display label
            sf_part   = m['label'].split(' / ')[0]
            model_part = m['label'].split(' / ', 1)[1] if ' / ' in m['label'] else m['label']
            display_label = f"{sf_labels.get(sf_part, sf_part)} / {model_part}"
            try:
                snr_m, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
                results.append({'label': display_label, 'snr': snr_m, 'ber': ber_m})
            except Exception as ex:
                errors.append(f"{display_label}:\n  {ex}")

        if errors:
            messagebox.showwarning('Load warnings',
                                   'Some models could not be loaded:\n\n' + '\n\n'.join(errors))
        if not results:
            messagebox.showinfo('Done', 'No models loaded successfully.')
            return

        save_path = self._var_save.get().strip() or None
        main_bers = self._var_load_main_ber(snr_targets)
        plot_variation_all(results, main_bers=main_bers, save_path=save_path, show=self._var_show.get())

    def _var_run_envelope(self):
        """For each sub-folder plot the worst/best BER envelope with filled area."""
        snr_targets, active_sf = self._var_get_snr_and_active_sf()
        if snr_targets is None:
            return

        folder = self._var_folder_var.get().strip()
        folder_results, errors = [], []

        for sf_name, sf_display in active_sf:
            models = discover_models_in_subfolders(folder, [sf_name])
            if not models:
                errors.append(f'{sf_name}: no valid models found')
                continue

            ber_rows = []
            for m in models:
                try:
                    _, ber_m = load_variation_ber(m['outputs_path'], snr_targets)
                    ber_rows.append(ber_m)
                except Exception as ex:
                    errors.append(f"{m['label']}:\n  {ex}")

            if not ber_rows:
                continue

            ber_mat = np.vstack(ber_rows)          # [N_models, N_snr]
            snr_rep, _ = load_variation_ber(models[0]['outputs_path'], snr_targets)

            folder_results.append({
                'label':   sf_display,
                'snr':     snr_rep,
                'ber_mat': ber_mat,
            })
            print(f'  {sf_display} ({sf_name}): {len(ber_rows)} model(s) loaded')

        if errors:
            messagebox.showwarning('Load warnings',
                                   'Some models could not be loaded:\n\n' + '\n\n'.join(errors))
        if not folder_results:
            messagebox.showinfo('Done', 'No data to plot.')
            return

        save_path = self._var_save.get().strip() or None
        main_bers = self._var_load_main_ber(snr_targets)
        plot_variation_envelope(folder_results, main_bers=main_bers, save_path=save_path, show=self._var_show.get())

            # ── shared helpers ────────────────────────────────────────────────────────

    def _browse_save(self, var):
        p = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'), ('All files', '*.*')],
            title='Save plot as…')
        if p:
            var.set(p)


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Architecture & Constellation
    # ══════════════════════════════════════════════════════════════════════════

    def _build_arch_tab(self, parent):
        pad = dict(padx=8, pady=4)

        # ── model selector ────────────────────────────────────────────────────
        lf_model = ctk.CTkFrame(parent)
        lf_model.grid(row=0, column=0, columnspan=2, sticky='ew', **pad)
        ctk.CTkLabel(lf_model, text='Model', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2), sticky='w')
        lf_model.columnconfigure(1, weight=1)

        ctk.CTkLabel(lf_model, text='Folder:').grid(row=1, column=0, sticky='w', **pad)
        self._arch_folder_var = tk.StringVar()
        self._arch_folder_entry = ctk.CTkEntry(lf_model, textvariable=self._arch_folder_var,
                                               width=400, state='readonly')
        self._arch_folder_entry.grid(row=1, column=1, sticky='ew', **pad)
        ctk.CTkButton(lf_model, text='Browse…', width=80,
                      command=self._arch_browse).grid(row=1, column=2, **pad)

        ctk.CTkLabel(lf_model, text='Stage:').grid(row=2, column=0, sticky='w', **pad)
        self._arch_stage_var = tk.IntVar(value=3)
        sf = ctk.CTkFrame(lf_model, fg_color='transparent')
        sf.grid(row=2, column=1, sticky='w')
        for s in (1, 2, 3):
            ctk.CTkRadioButton(sf, text=str(s), variable=self._arch_stage_var,
                               value=s).pack(side='left', padx=4)

        # ── constellation settings ────────────────────────────────────────────
        lf_const = ctk.CTkFrame(parent)
        lf_const.grid(row=1, column=0, sticky='nsew', **pad)
        ctk.CTkLabel(lf_const, text='Constellation settings',
                     font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(6,2))

        def _erow(lf, label, default, r):
            ctk.CTkLabel(lf, text=label, anchor='w').grid(row=r, column=0, sticky='w', **pad)
            v = tk.StringVar(value=default)
            ctk.CTkEntry(lf, textvariable=v, width=80).grid(row=r, column=1, **pad)
            return v

        self._arch_snr = _erow(lf_const, 'SNR (dB)',              '10', 1)
        self._arch_itr = _erow(lf_const, 'Noise avg iterations',  '1',  2)

        # ── output ────────────────────────────────────────────────────────────
        lf_out = ctk.CTkFrame(parent)
        lf_out.grid(row=1, column=1, sticky='nsew', **pad)
        ctk.CTkLabel(lf_out, text='Output', font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=0, column=0, columnspan=3, padx=8, pady=(6,2))
        ctk.CTkLabel(lf_out, text='Save plot to (optional)', anchor='w').grid(
            row=1, column=0, sticky='w', **pad)
        self._arch_save = tk.StringVar()
        ctk.CTkEntry(lf_out, textvariable=self._arch_save, width=220).grid(row=1, column=1, **pad)
        ctk.CTkButton(lf_out, text='Browse…', width=80,
                      command=lambda: self._browse_save(self._arch_save)).grid(row=1, column=2, **pad)
        self._arch_show = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(lf_out, text='Show plot interactively',
                        variable=self._arch_show, onvalue=True, offvalue=False).grid(
            row=2, column=0, columnspan=3, sticky='w', **pad)

        # ── run buttons ───────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(parent, fg_color='transparent')
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ctk.CTkButton(btn_frame, text='▶  Plot Architecture',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#8E44AD', hover_color='#6c3483',
                      command=self._arch_run_topology).grid(row=0, column=0, sticky='ew', padx=(0, 4))
        ctk.CTkButton(btn_frame, text='▶  Plot Constellation',
                      font=ctk.CTkFont(size=13, weight='bold'),
                      fg_color='#E67E22', hover_color='#b9600e',
                      command=self._arch_run_constellation).grid(row=0, column=1, sticky='ew', padx=(4, 0))

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    # ── arch helpers ──────────────────────────────────────────────────────────

    def _arch_browse(self):
        path = filedialog.askdirectory(title='Select model folder')
        if path:
            self._arch_folder_entry.configure(state='normal')
            self._arch_folder_var.set(path)
            self._arch_folder_entry.configure(state='readonly')

    def _arch_load(self):
        path = self._arch_folder_var.get().strip()
        if not path:
            messagebox.showwarning('No folder', 'Please select a model folder.')
            return None
        stage = self._arch_stage_var.get()
        try:
            model = load_model(path, stage)
            return model
        except Exception as e:
            messagebox.showerror('Load error', f'Failed to load model:\n{e}')
            return None

    def _arch_run_topology(self):
        model = self._arch_load()
        if model is None:
            return
        folder = self._arch_folder_var.get().strip()
        save_path = self._arch_save.get().strip() or None
        plot_architecture(model, folder_path=folder, save_path=save_path, show=self._arch_show.get())

    def _arch_run_constellation(self):
        model = self._arch_load()
        if model is None:
            return
        try:
            snr_db = float(self._arch_snr.get())
            n_itr  = int(self._arch_itr.get())
        except ValueError as e:
            messagebox.showerror('Invalid setting', str(e))
            return
        save_path = self._arch_save.get().strip() or None
        plot_constellation(model, snr_db=snr_db, n_itr=n_itr,
                           save_path=save_path, show=self._arch_show.get())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = App()
    app.mainloop()