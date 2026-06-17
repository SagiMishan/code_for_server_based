
"""
banch_marks.py
==============
For each SNR point, independently selects the best relay per channel
(the relay that maximises the worst-user SNR), then evaluates BER using
a simple hard-decision BPSK demodulator.

SNR formula used (matches your original code exactly):
    SNR_mn = (1/sigma2) * |w * g_mn * h_n|^2 / (1 + |w * g_mn|^2)
where
    w      = sqrt(P_relay / (sigma2 + |h_n|^2))   optimal linear gain
    h_n    = MatcgSR[c][0, n]                      source -> relay n
    g_mn   = cgRU[c][n, m]                         relay n -> user m
    sigma2 = P_source / SNR_lin                    noise variance
"""
import sys
from pathlib import Path
# Add the main folder (parent of code_compare_results, etc.) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from code_Networks.SmallFunctions import dB2lin

# ─────────────────────────────────────────────────────────────────────────────
# <<< USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

Name_of_model = "/home/dsi/mishans1/projects/code_for_server_based/compare_networks"

P_source = 1.0
P_relay  = 1.0

SNR_dB_vec = torch.arange(-20, 60, 0.5, dtype=torch.float32)   # -10 … 20 dB


# ─────────────────────────────────────────────────────────────────────────────
# load channel tensors
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# relay selection — per channel, per SNR point
#
#   resulted_SNR[m, n] = (1/sigma2) * |w_n * g_mn * h_n|^2
#                        ──────────────────────────────────
#                             1  +  |w_n * g_mn|^2
#
#   best relay n* = argmax_n  min_m  SNR[m, n]   (max-min over users)
# ─────────────────────────────────────────────────────────────────────────────

def select_best_relay(c: int, sigma2: float):
    """
    Returns (best_relay_idx, achieved_worst_user_SNR_tensor [N_users_c]).
    Uses your original SNR formula exactly.
    """
    h2 = MatcgSR[c].abs() ** 2                          # [1, N_relays]

    # optimal gain per relay  [N_users_c, N_relays]  (broadcast)
    w = torch.sqrt(
        torch.tensor(P_relay, dtype=torch.float32) /
        (sigma2 + h2)
    ).repeat((1,N_users[c]))

    # relay-to-user gains  [N_users_c, N_relays]
    g = cgRU[c]                                      # [N_users_c, N_relays]

    # source-to-relay gains broadcast to [N_users_c, N_relays]
    h = MatcgSR[c].repeat((1,N_users[c]))

    # SNR per (user, relay)
    wgh2 = (w * g * h).abs() ** 2
    wg2  = (w * g).abs() ** 2
    snr_matrix = (1.0 / sigma2) * wgh2 / (1.0 + wg2)   # [N_users_c, N_relays]

    # only consider relays visible to channel c
    # after — slice to N_relays columns only, dropping the source column
              # [N_users_c, K]

    worst_per_relay = snr_matrix.min(dim=1).values  # [N_relays]
    best_relay = worst_per_relay.argmax().item()  # global relay index directly
    snr_achieved = snr_matrix[best_relay,: ]  # [N_users_c]
    return best_relay, snr_achieved

# ─────────────────────────────────────────────────────────────────────────────
# BER from SNR — simple hard-decision BPSK demodulator
#   BER = Q(sqrt(2 * SNR)) = 0.5 * erfc(sqrt(SNR))
# ─────────────────────────────────────────────────────────────────────────────

def ber_bpsk(snr_lin: torch.Tensor) -> torch.Tensor:
    snr_np = snr_lin.detach().cpu().numpy()
    from scipy.special import erfc
    return torch.tensor(0.5 * erfc(np.sqrt(np.clip(snr_np, 0, None))),
                        dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# direct link SNR — source → user, no relay
#   SNR_direct_m = P_source * |cgTU[c][m]|^2 / sigma2
# ─────────────────────────────────────────────────────────────────────────────

def direct_link_snr(c: int, sigma2: float) -> torch.Tensor:
    """Returns SNR for every user on channel c via the direct link. [N_users_c]"""
    return P_source * cgTU[c].abs() ** 2 / sigma2


# ─────────────────────────────────────────────────────────────────────────────
# sweep over SNR vector
# ─────────────────────────────────────────────────────────────────────────────
for idx in range(10):
    path          = os.path.join(".", f"{Name_of_model}_{idx}", "")

    MatcgSR          = torch.load(os.path.join(os.pardir, path + "data","MatcgSR.pt"),          weights_only=True)
    cgRU             = torch.load(os.path.join(path + "data","cgRU.pt"),             weights_only=True)
    cgTU             = torch.load(os.path.join(path + "data","cgTU.pt"),             weights_only=True)
    connectaionMatrix= torch.load(os.path.join(path + "data","connectaionMatrix.pt"),weights_only=True)
    N_users          = torch.load(os.path.join(path + "data","N_users.pt"),          weights_only=True)
    N_channels       = int(torch.load(os.path.join(path + "data","N_channels.pt"),   weights_only=True))

    N_relays = MatcgSR[0].shape[1]     # inferred from tensor shape

    print(f"N_channels={N_channels}  N_relays={N_relays}  "
        f"N_users={N_users}")

    S = len(SNR_dB_vec)

    # [N_channels, S]  — worst-user BER per channel per SNR point
    worst_BER      = torch.zeros(N_channels, S)
    # [N_channels, S]  — which relay was chosen
    chosen_relay   = torch.zeros(N_channels, S, dtype=torch.long)
    # [N_channels, S]  — worst-user BER for direct source→user link
    direct_BER     = torch.zeros(N_channels, S)

    print(f"\n{'─'*70}")
    print(f"{'SNR(dB)':>8}  {'ch':>3}  {'relay':>6}  "
        f"{'worst SNR(dB)':>14}  {'worst BER':>10}")
    print(f"{'─'*70}")

    for si, snr_dB in enumerate(SNR_dB_vec):
        sigma2 = P_source / dB2lin(float(snr_dB))

        for c in range(N_channels):
            relay, snr_achieved = select_best_relay(c, float(sigma2))

            # worst user = the user with the lowest SNR
            worst_snr_lin  = snr_achieved.min()
            worst_snr_dB   = 10 * math.log10(max(float(worst_snr_lin), 1e-20))
            ber            = float(ber_bpsk(worst_snr_lin.unsqueeze(0))[0])

            worst_BER[c, si]    = ber
            chosen_relay[c, si] = relay

            # direct link — worst user SNR then BER
            d_snr             = direct_link_snr(c, float(sigma2))   # [N_users_c]
            d_worst_snr       = d_snr.min()
            d_ber             = float(ber_bpsk(d_worst_snr.unsqueeze(0))[0])
            direct_BER[c, si] = d_ber

            print(f"{float(snr_dB):>8.1f}  {c:>3}  {relay:>6}  "
                f"{worst_snr_dB:>14.2f}  {ber:>10.3e}  direct={d_ber:.3e}")

    # ─────────────────────────────────────────────────────────────────────────────
    # save results — one folder per benchmark, same layout as a trained model
    # ─────────────────────────────────────────────────────────────────────────────

    # worst performing channel across all channels
    overall_worst_BER = worst_BER.max(dim=0).values    # [S]
    direct_worst_BER  = direct_BER.max(dim=0).values   # [S]
    # relay chosen for the worst channel at each SNR point
    worst_ch_relay    = torch.zeros(S, dtype=torch.long)
    for si in range(S):
        worst_ch = int(worst_BER[:, si].argmax())
        worst_ch_relay[si] = chosen_relay[worst_ch, si]

    # ── benchmark relay ───────────────────────────────────────────────────────────
    relay_out = os.path.join( path , "Single relay per channel","Single relay per channel" )
    os.makedirs(os.path.join(relay_out , "outputs",""), exist_ok=True)
    torch.save(SNR_dB_vec,         os.path.join(relay_out , "outputs","SNR.pt"))
    torch.save(overall_worst_BER,  os.path.join(relay_out , "outputs","worst_BER.pt"))
    torch.save(worst_ch_relay,     os.path.join(relay_out , "outputs","relay_chosen.pt"))
    print(f"Relay benchmark saved  → {relay_out}outputs\\")

    # ── benchmark direct link ─────────────────────────────────────────────────────
    direct_out = os.path.join( path , "Direct link","Direct link" )
    os.makedirs(os.path.join(direct_out , "outputs",""), exist_ok=True)
    torch.save(SNR_dB_vec,        os.path.join(direct_out , "outputs","SNR.pt"))
    torch.save(direct_worst_BER,  os.path.join(direct_out , "outputs","worst_BER.pt"))
    print(f"Direct link benchmark saved → {direct_out}outputs\\")

    # ─────────────────────────────────────────────────────────────────────────────
    # BER vs SNR plot — worst channel only, relay=blue, direct=red
    # ─────────────────────────────────────────────────────────────────────────────

    snr_np = SNR_dB_vec.numpy()
    marks  = ['o', '^', 's', 'D', 'P', '*']

    plt.figure(figsize=(10, 6))
    plt.style.use('classic')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size']   = 12

    plt.title('BER vs SNR — worst channel benchmark\n'
            '(relay re-selected per SNR point  |  simple BPSK demodulator)',
            fontsize=13)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER (log scale)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.semilogy(snr_np, overall_worst_BER.numpy(),
                color='blue', linestyle='-', marker='o',
                linewidth=2.5, markersize=7,
                label="Best relay per channel (benchmark)")

    plt.semilogy(snr_np, direct_worst_BER.numpy(),
                color='red', linestyle='--', marker='^',
                linewidth=2.0, markersize=7,
                label="Direct link (no relay)")

    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(path, "Single relay per channel", "Single relay per channel", "outputs", "BER_vs_SNR.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved → {plot_path}")