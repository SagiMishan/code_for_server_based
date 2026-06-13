import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from Network_multy_channels import Network_multy_channel,load_model


def norm(input, p=2):
    return (input.abs() ** p) ** (1 / p)


BCEWithLogitsLoss_func = nn.BCEWithLogitsLoss(reduction='none')


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softwmax(LL):
    alpha = 5
    alpha = 5
    # Subtract max for numerical stability (log-sum-exp trick).
    # exp is computed once and reused instead of twice.
    LL_shifted = LL - LL.max().detach()
    e = torch.exp(alpha * LL_shifted)
    return (e * LL).sum() / e.sum()


def BCEWithLogitsLoss(pred, ground_truth):
    beta = 5
    yground_truth_nt = (ground_truth + 1) / 2
    LL = torch.mean(BCEWithLogitsLoss_func(beta * pred, yground_truth_nt), 0)
    return LL


def Single_channel_loss_function(pred, ground_truth):
    return softwmax(BCEWithLogitsLoss(pred, ground_truth))


def Multy_channel_loss_function(pred, ground_truth, V, N_channels):
    sum_loss = torch.tensor(0)
    for idx in range(N_channels):
        sum_loss = sum_loss + Single_channel_loss_function(pred[idx], ground_truth[idx] * 2 - 1)
    sum_loss = sum_loss + V
    return sum_loss


def randomComplexNormal(shape, sigma=1, mu=0):
    return sigma * torch.randn(shape, dtype=torch.complex64) + mu


def transmission_matrix(sources: torch.Tensor,
                        targets: torch.Tensor,
                        ref: torch.Tensor,
                        phi_deg: float,
                        eps: float = 1e-9) -> torch.Tensor:
    """
    sources: (Ns, 2) transmitters
    targets: (Nt, 2) candidate receivers
    ref    : (2,) reference point
    phi_deg: FULL sector angle (degrees). Sector is centered at (source - ref), i.e. away from ref.
    returns: (Ns, Nt) bool, True if target j lies in source i's TX sector
    """
    sources = sources.float()
    targets = targets.float()
    ref = ref.view(-1).float()

    Ns, Nt = sources.size(0), targets.size(0)
    cos_thr = math.cos(math.radians(phi_deg / 2.0))

    # TX sector centers for each source i: C_tx[i] = sources[i] - ref
    C_tx = sources - ref  # (Ns, 2)
    C_tx_norm = torch.linalg.norm(C_tx, dim=1, keepdim=True)
    C_tx_u = torch.where(C_tx_norm > eps, C_tx / C_tx_norm, torch.zeros_like(C_tx))  # (Ns, 2)

    # Pairwise direction from i->j: V[i,j] = targets[j] - sources[i]
    V = targets.unsqueeze(0) - sources.unsqueeze(1)  # (Ns, Nt, 2)
    V_norm = torch.linalg.norm(V, dim=2, keepdim=True)  # (Ns, Nt, 1)
    V_u = torch.where(V_norm > eps, V / V_norm, torch.zeros_like(V))  # (Ns, Nt, 2)

    # Angle test via cosine
    dots = (V_u * C_tx_u.unsqueeze(1)).sum(dim=2).clamp(-1.0, 1.0)  # (Ns, Nt)
    return dots >= cos_thr


def reception_matrix(sources: torch.Tensor,
                     targets: torch.Tensor,
                     ref: torch.Tensor,
                     phi_deg: float,
                     eps: float = 1e-9) -> torch.Tensor:
    """
    sources: (Ns, 2) transmitters
    targets: (Nt, 2) receivers being tested
    ref    : (2,) reference point
    phi_deg: FULL sector angle (degrees). RX sector is centered at (ref - target), i.e. toward ref.
    returns: (Ns, Nt) bool, True if target j can RX a signal arriving from source i
    """
    sources = sources.float()
    targets = targets.float()
    ref = ref.view(-1).float()

    Ns, Nt = sources.size(0), targets.size(0)
    cos_thr = math.cos(math.radians(phi_deg / 2.0))

    # RX sector centers for each target j: T_rx[j] = ref - targets[j]
    T_rx = ref - targets  # (Nt, 2)
    T_rx_norm = torch.linalg.norm(T_rx, dim=1, keepdim=True)
    T_rx_u = torch.where(T_rx_norm > eps, T_rx / T_rx_norm, torch.zeros_like(T_rx))  # (Nt, 2)

    # Arrival direction at target j from source i is (source - target) = - (targets - sources)
    V = targets.unsqueeze(0) - sources.unsqueeze(1)  # (Ns, Nt, 2)
    A = -V  # (Ns, Nt, 2)
    A_norm = torch.linalg.norm(A, dim=2, keepdim=True)  # (Ns, Nt, 1)
    A_u = torch.where(A_norm > eps, A / A_norm, torch.zeros_like(A))  # (Ns, Nt, 2)

    # Angle test via cosine
    dots = (A_u * T_rx_u.unsqueeze(0)).sum(dim=2).clamp(-1.0, 1.0)  # (Ns, Nt)
    return dots >= cos_thr


# def crateNetworkConnaction(path, N_transmitter, N_relays, N_users, N_channels=2, alpha=4, R=1, Phi=45, sector=60,
#                            R_user=1.1, N_ant=1, freq=2.4e9):
#     """
#     N_ant : number of antennas per user (ULA placed on the circumference of the user circle)
#     freq  : carrier frequency [Hz] — used to compute lambda = c/freq, element spacing = lambda/2
#
#     Antenna array geometry
#     ----------------------
#     Each user sits at angle theta_u on the circle of radius R_user.
#     The tangent direction at that point is (-sin(theta_u), cos(theta_u)).
#     The N_ant antenna elements are placed along this tangent, centred on the
#     user position, with inter-element spacing d = lambda/2:
#         ant_pos[u, n] = user_pos[u] + (n - (N_ant-1)/2) * d * tangent[u],  n = 0...N_ant-1
#
#     Channel gains to a user with N_ant antennas gain an extra antenna dimension:
#         cgRU : list[c] of (N_relays, N_users[c], N_ant)   -- relay to user antennas
#         cgTU : list[c] of (N_users[c], N_ant)             -- TX to user antennas
#     """
#     c_light = 3e8  # speed of light [m/s]
#     lam = c_light / freq  # wavelength [m]
#     d_ant = lam / 2  # inter-element spacing [m]
#
#     # ── 1. Relay positions: uniform random inside disk of radius R ────────────
#     r = R * torch.sqrt(torch.rand(N_relays))
#     phi_relay = 2 * torch.pi * torch.rand(N_relays)
#     relay_x = r * torch.cos(phi_relay)
#     relay_y = r * torch.sin(phi_relay)
#     points = torch.stack((relay_x, relay_y), dim=1)
#     posR = points  # (N_relays, 2)
#     dists_RR = torch.cdist(posR, posR)  # (N_relays, N_relays)
#
#     # ── 2. Transmitter positions: equally spaced on circle of radius R_user ──
#     # Spacing = 2*pi / N_transmitter  ->  maximally far apart from each other.
#     phi_trans = 2 * torch.pi * torch.arange(N_transmitter) / N_transmitter
#     trans_x = R_user * torch.cos(phi_trans)
#     trans_y = R_user * torch.sin(phi_trans)
#     trans_points = torch.stack((trans_x, trans_y), dim=1)  # (N_transmitter, 2)
#
#     # ── 3. Receiver (user) positions + per-antenna positions ─────────────────
#     # The sector angular width is given by the `sector` input parameter.
#     # Receivers are spaced uniformly within that sector, centred opposite
#     # the TX (i.e. shifted by pi from the TX angle).
#     sector_rad = np.deg2rad(sector)
#     half_sector = sector_rad / 2
#
#     # Antenna element offsets along the tangent: centred at 0, step = lambda/2
#     # The straight-line (Cartesian) distance between adjacent elements is lambda/2.
#     ant_idx = torch.arange(N_ant) - (N_ant - 1) / 2.0  # (N_ant,)  e.g. -1, 0, 1 for N_ant=3
#     ant_offsets = ant_idx * d_ant  # (N_ant,)  in metres
#
#     user_point = []  # list[c] of (N_users[c], 2)         -- user centres
#     user_ant_point = []  # list[c] of (N_users[c], N_ant, 2)  -- antenna positions
#
#     for c in range(N_channels):
#         n_u = N_users[c]
#         if n_u == 1:
#             offsets = torch.tensor([0.0])
#         else:
#             offsets = torch.linspace(-half_sector, half_sector, n_u)
#
#         # User centre angles (opposite side of circle from TX)
#         base_angle = phi_trans[c] + np.pi + offsets  # (n_u,)
#         u_x = R_user * torch.cos(base_angle)  # (n_u,)
#         u_y = R_user * torch.sin(base_angle)  # (n_u,)
#         centres = torch.stack((u_x, u_y), dim=1)  # (n_u, 2)
#         user_point.append(centres)
#
#         # Tangent direction at each user position: (-sin(theta), cos(theta))
#         # This is the unit vector along the circumference at that point.
#         tangent = torch.stack((-torch.sin(base_angle),
#                                torch.cos(base_angle)), dim=1)  # (n_u, 2)
#
#         # Antenna positions in Cartesian coordinates:
#         #   ant_pos[u, i] = centre[u] + ant_offsets[i] * tangent[u]
#         # Straight-line distance between adjacent elements = d_ant = lambda/2
#         # centres:     (n_u, 2)  -> (n_u, 1, 2)
#         # ant_offsets: (N_ant,)  -> (1, N_ant, 1)
#         # tangent:     (n_u, 2)  -> (n_u, 1, 2)
#         ant_pos = centres.unsqueeze(1) + ant_offsets.view(1, N_ant, 1) * tangent.unsqueeze(1)
#         # shape: (n_u, N_ant, 2)
#         user_ant_point.append(ant_pos)
#
#     posU = user_point
#
#     # ── 4. Build channel matrices per channel ─────────────────────────────────
#     all_connectaionMatrix = []
#     MatcgSR = []
#     MatcgRR = []
#     cgRU = []  # list[c] of (N_relays, N_users[c], N_ant)
#     cgTU = []  # list[c] of (N_users[c], N_ant)
#
#     for c in range(N_channels):
#         connectaionMatrix = torch.zeros((N_relays + 1, N_relays + 1))
#
#         # ── Relay-to-relay connectivity (directional antenna masks) ───────────
#         transmit_to = transmission_matrix(posR, posR, trans_points[c], Phi)
#         recive_from = reception_matrix(posR, posR, trans_points[c], Phi)
#         links = transmit_to & recive_from
#         links.fill_diagonal_(False)
#         connectaionMatrix[:N_relays, :N_relays] = links * 1
#
#         # ── Relay-to-relay channel gains ──────────────────────────────────────
#         v = randomComplexNormal([dists_RR.shape[0], dists_RR.shape[1]])
#         cg_RR = connectaionMatrix[:N_relays, :N_relays] * v * torch.pow(
#             dists_RR, torch.Tensor([-alpha / 2]))
#         MatcgRR.append(cg_RR)
#
#         # ── TX-to-relay connectivity & channel gains ──────────────────────────
#         TO = -trans_points[c]
#         TR = posR - trans_points[c]
#         cos_tx = ((TO * TR).sum(dim=1) / (
#                 torch.norm(TR, dim=1) * torch.norm(TO)
#         )).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
#
#         connectaionMatrix[-1, :N_relays] = cos_tx >= torch.cos(torch.tensor(np.deg2rad(Phi / 2)))
#         dists_SR = torch.cdist(points, trans_points[c].unsqueeze(0))
#         v = randomComplexNormal([N_relays, 1])
#         cg_SR = connectaionMatrix[-1, :N_relays].unsqueeze(1) * v * torch.pow(
#             dists_SR, torch.Tensor([-alpha / 2]))
#         MatcgSR.append(cg_SR.T)
#         all_connectaionMatrix.append(connectaionMatrix)
#
#         # ── Relay-to-user connectivity & channel gains per antenna ──────────────
#         # Connectivity is tested per antenna element:
#         #   relay n can reach antenna (u, a) if the arrival angle is within Phi.
#         # A user is reachable if at least one of its antennas passes the test.
#         ant_pos_c = user_ant_point[c]  # (N_users[c], N_ant, 2)
#         ant_pos_flat = ant_pos_c.reshape(-1, 2)  # (N_users[c]*N_ant, 2)
#
#         # Build per-antenna connectivity mask: (N_relays, N_users[c]*N_ant)
#         connectaionMatrix_ant = torch.zeros(N_relays, N_users[c] * N_ant)
#         for idx in range(N_users[c]):
#             for a in range(N_ant):
#                 ant_xy = ant_pos_c[idx, a]  # (2,)  this antenna's position
#                 p2O = ant_xy - trans_points[c]  # direction: antenna -> TX origin
#                 p2p1 = ant_xy - posR  # direction: antenna -> each relay (N_relays, 2)
#                 dot_rx = (p2p1 * p2O).sum(dim=1) / (
#                         1e-6 + torch.norm(p2p1, dim=1) * torch.norm(p2O))
#                 dot_rx = dot_rx.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
#                 col = idx * N_ant + a
#                 connectaionMatrix_ant[:, col] = dot_rx >= torch.cos(torch.tensor(Phi / 2))
#
#         # Check every user has at least one reachable antenna
#         # Sum over the N_ant columns belonging to each user
#         ant_mask_per_user = connectaionMatrix_ant.view(N_relays, N_users[c], N_ant)
#         reachable = ant_mask_per_user.sum(dim=0).sum(dim=1)  # (N_users[c],)
#         if torch.any(reachable == 0):
#             raise Exception("Sorry, for one of users signal canot be reach , try again")
#
#         # ── Relay-to-user channel gains: (N_relays, N_users[c]*N_ant) ───────────
#         dists_RU_flat = torch.cdist(points, ant_pos_flat)  # (N_relays, N_users[c]*N_ant)
#         v_RU = randomComplexNormal([N_relays, N_users[c] * N_ant])
#         cgRU.append(connectaionMatrix_ant * v_RU * torch.pow(dists_RU_flat, -alpha / 2))
#         # shape: (N_relays, N_users[c]*N_ant)
#
#         # ── TX-to-user channel gains: (N_users[c]*N_ant,) ────────────────────
#         tx_pos_c = trans_points[c].unsqueeze(0)  # (1, 2)
#         dists_TU_flat = torch.cdist(tx_pos_c, ant_pos_flat)  # (1, N_users[c]*N_ant)
#         v_TU = randomComplexNormal([1, N_users[c] * N_ant])
#         cgTU.append((v_TU * torch.pow(dists_TU_flat, -alpha / 2)).squeeze(0))
#         # shape: (N_users[c]*N_ant,)
#
#         # ── Visualisation ─────────────────────────────────────────────────────
#         cmap = plt.get_cmap("tab10")
#         color = cmap(c % cmap.N)
#         plt.scatter(trans_points[c, 0], trans_points[c, 1],
#                     marker="$T$", s=200, color=color)
#         plt.scatter(posU[c][:, 0], posU[c][:, 1],
#                     marker="$R$", s=200, color=color)
#         # Show individual antenna element positions
#         ant_flat = ant_pos_c.reshape(-1, 2)
#         plt.scatter(ant_flat[:, 0], ant_flat[:, 1],
#                     marker="+", s=60, color=color, linewidths=0.8)
#
#     color = cmap((N_channels + 1) % cmap.N)
#     plt.scatter(posR[:, 0], posR[:, 1], marker="o", s=100, color=color)
#     plt.savefig(path + "blank_architecture.png")
#     plt.show()
#
#     return all_connectaionMatrix, MatcgSR, MatcgRR, cgRU, cgTU, posR, posU, trans_points
#

def crateNetworkConnaction(path, N_transmitter, N_relays, N_users, N_channels=2, alpha=4, R=1, Phi=45, sector=60,
                           R_user=1.1, N_rx=2, N_tx=1, d_spacing=0.01):
    """
    N_rx  : number of receive antennas per user (ULA along the tangent of the user circle)
    N_tx  : number of transmit antennas per transmitter (ULA along the tangent of the TX circle)
    d_spacing : inter-element antenna spacing in metres (default 0.1 m = 10 cm)

    Antenna array geometry
    ----------------------
    Both TX and RX arrays are placed along the tangent of the circle at their position,
    centred on the node, with inter-element Cartesian spacing d = d_spacing:
        ant_pos[n] = centre + (n - (N-1)/2) * d_spacing * tangent,  n = 0 ... N-1

    Channel gain shapes:
        MatcgSR : list[c] of (N_relays, N_tx)               -- TX antennas to relay
        cgRU    : list[c] of (N_relays, N_users[c]*N_rx)    -- relay to RX antennas
        cgTU    : list[c] of (N_tx, N_users[c]*N_rx)        -- TX antennas to RX antennas
    """
    d_ant = d_spacing  # inter-element spacing [m]

    # ── 1. Relay positions: uniform random inside disk of radius R ────────────
    r = R * torch.sqrt(torch.rand(N_relays))
    phi_relay = 2 * torch.pi * torch.rand(N_relays)
    relay_x = r * torch.cos(phi_relay)
    relay_y = r * torch.sin(phi_relay)
    points = torch.stack((relay_x, relay_y), dim=1)
    posR = points  # (N_relays, 2)
    dists_RR = torch.cdist(posR, posR)  # (N_relays, N_relays)

    # ── 2. Transmitter positions: equally spaced on circle of radius R_user ──
    # Spacing = 2*pi / N_transmitter  ->  maximally far apart from each other.
    phi_trans = 2 * torch.pi * torch.arange(N_transmitter) / N_transmitter
    trans_x = R_user * torch.cos(phi_trans)
    trans_y = R_user * torch.sin(phi_trans)
    trans_points = torch.stack((trans_x, trans_y), dim=1)  # (N_transmitter, 2)

    # ── 3. Receiver (user) positions + per-antenna positions ─────────────────
    # The sector angular width is given by the `sector` input parameter.
    # Receivers are spaced uniformly within that sector, centred opposite
    # the TX (i.e. shifted by pi from the TX angle).
    sector_rad = np.deg2rad(sector)
    half_sector = sector_rad / 2

    # ── RX antenna offsets (per user) ────────────────────────────────────────
    # Centred at 0, step = d_spacing, along the tangent of the user circle.
    rx_idx = torch.arange(N_rx) - (N_rx - 1) / 2.0  # (N_rx,)
    rx_offsets = rx_idx * d_ant  # (N_rx,)  metres

    # ── TX antenna offsets (per transmitter) ─────────────────────────────────
    tx_idx = torch.arange(N_tx) - (N_tx - 1) / 2.0  # (N_tx,)
    tx_offsets = tx_idx * d_ant  # (N_tx,)  metres

    user_point = []  # list[c] of (N_users[c], 2)          -- user centres
    user_ant_point = []  # list[c] of (N_users[c], N_rx, 2)    -- RX antenna positions

    for c in range(N_channels):
        n_u = N_users[c]
        if n_u == 1:
            offsets = torch.tensor([0.0])
        else:
            offsets = torch.linspace(-half_sector, half_sector, n_u)

        # User centre angles (opposite side of circle from TX)
        base_angle = phi_trans[c] + np.pi + offsets  # (n_u,)
        u_x = R_user * torch.cos(base_angle)  # (n_u,)
        u_y = R_user * torch.sin(base_angle)  # (n_u,)
        centres = torch.stack((u_x, u_y), dim=1)  # (n_u, 2)
        user_point.append(centres)

        # RX tangent direction at each user position: (-sin(theta), cos(theta))
        rx_tangent = torch.stack((-torch.sin(base_angle),
                                  torch.cos(base_angle)), dim=1)  # (n_u, 2)

        # RX antenna positions: ant_pos[u, i] = centre[u] + rx_offsets[i] * rx_tangent[u]
        # centres:     (n_u, 2)  -> (n_u, 1, 2)
        # rx_offsets:  (N_rx,)   -> (1, N_rx, 1)
        # rx_tangent:  (n_u, 2)  -> (n_u, 1, 2)
        ant_pos = centres.unsqueeze(1) + rx_offsets.view(1, N_rx, 1) * rx_tangent.unsqueeze(1)
        # shape: (n_u, N_rx, 2)
        user_ant_point.append(ant_pos)

    posU = user_point

    # ── 3b. TX antenna positions per transmitter ─────────────────────────────
    # Each TX also has an array along the tangent of the TX circle.
    # tx_ant_point[c]: (N_tx, 2)
    tx_ant_point = []
    for c in range(N_channels):
        tx_tangent = torch.stack((-torch.sin(phi_trans[c:c + 1]),
                                  torch.cos(phi_trans[c:c + 1])), dim=1)  # (1, 2)
        tx_ant_pos = trans_points[c].unsqueeze(0) + tx_offsets.view(N_tx, 1) * tx_tangent  # (N_tx, 2)
        tx_ant_point.append(tx_ant_pos)

    # ── 4. Build channel matrices per channel ─────────────────────────────────
    all_connectaionMatrix = []
    MatcgSR = []  # list[c] of (N_relays, N_tx)
    MatcgRR = []
    cgRU = []  # list[c] of (N_relays, N_users[c]*N_rx)
    cgTU = []  # list[c] of (N_tx, N_users[c]*N_rx)

    for c in range(N_channels):
        connectaionMatrix = torch.zeros((N_relays + 1, N_relays + 1))

        # ── Relay-to-relay connectivity (directional antenna masks) ───────────
        transmit_to = transmission_matrix(sources=posR, targets=posR, ref=trans_points[c], phi_deg=Phi)
        recive_from = reception_matrix(sources=posR, targets=posR, ref=trans_points[c], phi_deg=Phi)
        links = transmit_to & recive_from
        links.fill_diagonal_(False)
        connectaionMatrix[:N_relays, :N_relays] = links * 1

        # ── Relay-to-relay channel gains ──────────────────────────────────────
        v = randomComplexNormal([dists_RR.shape[0], dists_RR.shape[1]])
        cg_RR = connectaionMatrix[:N_relays, :N_relays] * v * torch.pow(
            dists_RR, torch.Tensor([-alpha / 2]))
        MatcgRR.append(cg_RR)

        # ── TX-to-relay connectivity & channel gains ──────────────────────────
        TO = -trans_points[c]
        TR = posR - trans_points[c]
        # dot_tx = torch.acos(
        #     (TO * TR).sum(dim=1) / (torch.norm(TR, dim=1) * torch.norm(TO))
        # ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # connectaionMatrix[-1, :N_relays] = dot_tx >= torch.cos(
        #     torch.tensor(np.deg2rad(sector/2)))
        cos_tx = ((TO * TR).sum(dim=1) /
                  (torch.norm(TR, dim=1) * torch.norm(TO) + 1e-9)
                  ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # Relay is inside the TX sector if angle <= sector/2
        # i.e. cos(angle) >= cos(sector/2)
        cos_half_sector = torch.cos(torch.tensor(np.deg2rad(sector / 2)))
        connectaionMatrix[-1, :N_relays] = cos_tx >= cos_half_sector

        # ── TX antenna to relay: (N_relays, N_tx) ───────────────────────────────
        # Distance from each TX antenna element to each relay.
        dists_SR = torch.cdist(points, tx_ant_point[c])  # (N_relays, N_tx)
        v_SR = randomComplexNormal([N_relays, N_tx])
        # Connectivity mask broadcast over TX antennas: (N_relays, 1)
        mask_SR = connectaionMatrix[-1, :N_relays].unsqueeze(1)
        cg_SR = mask_SR * v_SR * torch.pow(dists_SR, -alpha / 2)
        MatcgSR.append(cg_SR)  # (N_relays, N_tx)
        all_connectaionMatrix.append(connectaionMatrix)

        # ── Relay-to-user connectivity & channel gains per antenna ──────────────
        # Connectivity is tested per antenna element:
        #   relay n can reach antenna (u, a) if the arrival angle is within Phi.
        # A user is reachable if at least one of its antennas passes the test.
        ant_pos_c = user_ant_point[c]  # (N_users[c], N_rx, 2)
        ant_pos_flat = ant_pos_c.reshape(-1, 2)  # (N_users[c]*N_rx, 2)

        # Build per-RX-antenna connectivity mask: (N_relays, N_users[c]*N_rx)
        connectaionMatrix_ant = torch.zeros(N_relays, N_users[c] * N_rx)
        for idx in range(N_users[c]):
            for a in range(N_rx):
                ant_xy = ant_pos_c[idx, a]  # (2,)
                p2O = ant_xy - trans_points[c]  # direction: RX antenna -> TX centre
                p2p1 = ant_xy - posR  # direction: RX antenna -> each relay
                cos_rx = ((p2O * p2p1).sum(dim=1) /
                          (torch.norm(p2p1, dim=1) * torch.norm(p2O) + 1e-9)
                          ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                # Relay is inside the TX sector if angle <= sector/2
                # i.e. cos(angle) >= cos(sector/2)
                cos_half_sector = torch.cos(torch.tensor(np.deg2rad(Phi / 2)))
                col = idx * N_rx + a
                connectaionMatrix_ant[:, col] = cos_rx >= cos_half_sector

        # Check every user has at least one reachable RX antenna
        ant_mask_per_user = connectaionMatrix_ant.view(N_relays, N_users[c], N_rx)
        reachable = ant_mask_per_user.sum(dim=0).sum(dim=1)  # (N_users[c],)
        if torch.any(reachable == 0):
            raise Exception("Sorry, for one of users signal canot be reach , try again")

        # ── Relay-to-user channel gains: (N_relays, N_users[c]*N_rx) ────────────
        dists_RU_flat = torch.cdist(points, ant_pos_flat)  # (N_relays, N_users[c]*N_rx)
        v_RU = randomComplexNormal([N_relays, N_users[c] * N_rx])
        cgRU.append(connectaionMatrix_ant * v_RU * torch.pow(dists_RU_flat, -alpha / 2))
        # shape: (N_relays, N_users[c]*N_rx)

        # ── TX-to-user channel gains: (N_tx, N_users[c]*N_rx) ────────────────
        # Each TX antenna to each RX antenna.
        dists_TU_flat = torch.cdist(tx_ant_point[c], ant_pos_flat)  # (N_tx, N_users[c]*N_rx)
        v_TU = randomComplexNormal([N_tx, N_users[c] * N_rx])
        cgTU.append(v_TU * torch.pow(dists_TU_flat, -alpha / 2))
        # shape: (N_tx, N_users[c]*N_rx)

        # ── Visualisation ─────────────────────────────────────────────────────
        cmap = plt.get_cmap("tab10")
        color = cmap(c % cmap.N)
        plt.scatter(trans_points[c, 0], trans_points[c, 1],
                    marker="$T$", s=200, color=color)
        plt.scatter(posU[c][:, 0], posU[c][:, 1],
                    marker="$R$", s=200, color=color)
        # Show RX antenna element positions
        rx_flat = ant_pos_c.reshape(-1, 2)
        plt.scatter(rx_flat[:, 0], rx_flat[:, 1],
                    marker="+", s=60, color=color, linewidths=0.8)
        # Show TX antenna element positions
        plt.scatter(tx_ant_point[c][:, 0], tx_ant_point[c][:, 1],
                    marker="x", s=60, color=color, linewidths=0.8)

    color = cmap((N_channels + 1) % cmap.N)
    plt.scatter(posR[:, 0], posR[:, 1], marker="o", s=100, color=color)
    plt.savefig(os.path.join(path, "blank_architecture.png"))
    plt.clf()

    return all_connectaionMatrix, MatcgSR, MatcgRR, cgRU, cgTU, posR, posU, trans_points


def lin2dB(x):
    return 10 * torch.log10(torch.tensor(x))


def dB2lin(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return torch.pow(10, x / 10)


def plotSNRvsBER(model, batch, SNR, stage, num_itr=1):
    plt.close()
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.style.use('classic')
    # MATLAB-like color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    # Use Helvetica-like font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    plt.title('BER vs SNR - {}'.format(stage), fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER (log scale)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    worst_BER = torch.zeros((model.N_channels, SNR.shape[0]))
    best_BER = torch.zeros((model.N_channels, SNR.shape[0]))
    model.eval()
    for c in range(model.N_channels):

        for snr_idx, snr in enumerate(SNR):
            model.sub_networks[c].SNR = dB2lin(snr)
            for itr in range(num_itr):
                signal,bits = model.sub_networks[c].modulator(batch )
                # Compute prediction error
                rm = model.sub_networks[c](signal,bits)
                pred = model.sub_networks[c].demodulator(rm)
                scalar_worst_BER, _, scalar_best_BER = model.sub_networks[c].BER(bits=bits, pred=pred)
                # print(f"{signal.shape=}, {bits.shape=}, {rm.shape=}, {pred.shape=}")
                worst_BER[c, snr_idx] += scalar_worst_BER
                best_BER[c, snr_idx] += scalar_best_BER

            worst_BER[c, snr_idx] /= num_itr
            best_BER[c, snr_idx] /= num_itr

            print("channel {} -- In valdation: SNR: {},  worst BER : {:.2E}, best BER : {:.2E}".format(c, snr,
                                                                                                       worst_BER[
                                                                                                           c, snr_idx],
                                                                                                       best_BER[
                                                                                                           c, snr_idx]))
            if worst_BER[c, snr_idx] == 0:
                break
        plt.semilogy(SNR, worst_BER[c, :], '-o', label=f"channel {c} - worst")
        # plt.semilogy(SNR, best_BER[c, :], '-o', label=f"channel {c} - best")

    plt.legend(loc="lower left")
    return worst_BER, best_BER


def compere_all_stages(model, path, batch_size, SNR, num_itr=1):
    markers = ['-o', '-^', '-D']
    plt.figure(figsize=(10, 6))
    plt.style.use('classic')

    # MATLAB-like color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    # Use Helvetica-like font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    plt.title('Compare all the stages - worst BER ', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('BER (log scale)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    torch.save(SNR, os.path.join(path, "outputs", "SNR.pt"))

    with torch.no_grad():
        for stage in range(1, 4):
            model.load(path, f"stage_{stage}")
            Worst_ber = torch.zeros((model.N_channels, SNR.shape[0]))
            with torch.no_grad():
                for c in range(model.N_channels):
                    for snr_index, snr in enumerate(SNR):
                        model.sub_networks[c].SNR = dB2lin(snr)

                        for itr in range(num_itr):
                            signal,bits = model.sub_networks[c].modulator(batch_size=batch_size)
                            # Compute prediction error
                            rm = model.sub_networks[c](signal,bits)
                            pred = model.sub_networks[c].demodulator(rm)
                            scalar_worst_BER, _, scalar_best_BER = model.sub_networks[c].BER(bits=bits, pred=pred)
                            Worst_ber[c, snr_index] += scalar_worst_BER
                        Worst_ber[c, snr_index] = Worst_ber[c, snr_index] / num_itr
                        print(
                            "stage {} -channel {} -- In valdation: SNR: {},  worst BER : {:.2E}".format(stage, c, snr,
                                                                                                        Worst_ber[
                                                                                                            c, snr_index]))
                        if Worst_ber[c, snr_index] == 0:
                            break
                    torch.save(Worst_ber, os.path.join(path, "outputs", f"worst_BER_stage_{stage}.pt"))
                    plt.semilogy(SNR, Worst_ber[c, :], markers[stage - 1], label=f"stage{stage} - channel {c}")

    plt.legend(loc="lower left")
    plt.savefig(os.path.join(path, "compare_3_stages.png"))
    plt.clf()


def plot_architecture(path, Name_of_model = None, stage="stage_2"):

    posT = torch.load(os.path.join(path, "data", "posT.pt"), weights_only=False)
    posR = torch.load(os.path.join(path, "data", "posR.pt"), weights_only=False)
    posU = torch.load(os.path.join(path, "data", "posU.pt"), weights_only=False)


    model = load_model(path)
    if not Name_of_model is None:
        model.load(os.path.join(path, Name_of_model), stage)
    else:
        model.load(path, stage)
    N_channels = model.N_channels
    color_RR = np.zeros((len(posR), 1))

    model.update_v()
    model.culc_p()
    for c in range(N_channels):
        pl = model.P[c]
        color_RR[pl >= 0.5] = c + 1

    colors = plt.get_cmap('jet', N_channels + 1)
    plt.figure(figsize=(12, 10))
    total_relay = 0
    for c in range(N_channels):
        plt.scatter(posT[c, 0], posT[c, 1], marker="$T$", label=f"GS - channel {c}", color=colors(c), s=200)
        plt.scatter(posR[np.squeeze(color_RR == c + 1), 0], posR[np.squeeze(color_RR == c + 1), 1], marker='o',
                    color=colors(c),
                    label=f"relay - channel - {c}", s=200)
        plt.scatter(posU[c][:, 0], posU[c][:, 1],
                    marker="$R$", color=colors(c),
                    label=f"user - channel - {c}", s=200)
        total_relay+=posR[np.squeeze(color_RR == c + 1), 0].size()[0]
    if total_relay != model.N_relays:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%" * 100)
        print("heeeeeeeeeeeeeeeelllllllllllllllll nooooooooooooooooooooo")
        print(f"there is an error in model {os.path.join(path, Name_of_model)}! as stage {stage}")
        print("the total relays printed is not the same as model.N_relays")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%" * 100)
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncols=3 * N_channels, mode="expand", borderaxespad=0.)

    if not Name_of_model is None:
        plt.savefig(os.path.join(path, Name_of_model + "architecture.png"))
    else:
        plt.savefig(os.path.join(path, "architecture.png"))

    plt.clf()


def nested_list_to_tensor(nested_list):
    """
    Recursively convert a nested list of tensors into a PyTorch tensor.

    Args:
        nested_list: A nested list structure containing PyTorch tensors.

    Returns:
        A PyTorch tensor combining all the nested tensors.
    """
    # Base case: If the input is already a tensor, return it
    if isinstance(nested_list, torch.Tensor):
        return nested_list

    # Recursive case: Apply the function to all elements in the list
    stacked_list = [nested_list_to_tensor(sublist) for sublist in nested_list]

    # Stack along a new dimension
    return torch.stack(stacked_list)


def check_for_nan_inf(tensor, name="Tensor"):
    if torch.any(torch.isnan(tensor)):
        print(f"{name} contains NaN!")
        return True
    if torch.any(torch.isinf(tensor)):
        print(f"{name} contains Inf!")
        return True
    return False


import scipy.io


def save_var_with_name(var, name, prefix=''):
    if isinstance(var, list):
        for idx, elem in enumerate(var):
            new_name = f"{name}_{idx}"
            save_var_with_name(elem, new_name, prefix)
    else:
        filename = os.path.join(prefix, f"{name}.mat")
        scipy.io.savemat(filename, {name: var})


def int_to_binary(ints, num_bits):
    # Create a mask for each bit position (e.g., [128, 64, 32, ..., 1])
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(ints.device)

    # Use bitwise AND to check if each bit is set
    # unsqueeze(-1) adds a new dimension to allow broadcasting
    return (ints.unsqueeze(-1).bitwise_and(mask).ne(0)).to(torch.int)


def plot_symbols(model, path):
    fig, axes = plt.subplots(2, model.N_channels,
                             figsize=(8 * model.N_channels, 8),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for c in range(model.N_channels):
        ax_tx = axes[0, c]
        ax_rx = axes[1, c]

        N_symbols = 2 ** model.N_users[c]
        bits = int_to_binary(torch.arange(0, N_symbols), model.N_users[c]).to(torch.complex64)
        if model.demod_type == "complex":
            s = model.sub_networks[c].transmitNN(bits).detach()
        else:
            symbols_int = torch.sum(2 ** (torch.unsqueeze(torch.linspace(0, model.N_users[c] - 1, model.N_users[c]), dim=1)) * bits.T,
                            dim=0)
            s = model.sub_networks[c].modulation[symbols_int.real.to(torch.int32)]

        print(f"for channel {c} the maximum amplitude is {torch.max(s.abs() ** 2).item()}")
        rm = model.sub_networks[c](s,bits.T).detach().T
        if model.demod_type == "simple":
            s = torch.unsqueeze(s,dim=1)
        # bits label per symbol: e.g. "01" for symbol 1 with 2 users
        bits_labels = [
            "".join(str(b) for b in int_to_binary(torch.tensor([sym]), model.N_users[c])[0].int().tolist())
            for sym in range(N_symbols)
        ]

        # ── Transmitter output (top row) ──
        for idx in range(model.N_tx):
            color = cmap(idx)
            ax_tx.scatter(s[:, idx].real, s[:, idx].imag, label=f"Tx {idx}", zorder=3,color=color)
            for sym in range(N_symbols):
                ax_tx.annotate(
                    bits_labels[sym],
                    (s[sym, idx].real.item(), s[sym, idx].imag.item()),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, fontweight="bold"
                )
        ax_tx.set_title(f"Channel {c} — Transmitter output")
        ax_tx.set_xlabel("Real")
        ax_tx.set_ylabel("Imag")
        ax_tx.legend()
        ax_tx.grid(True, linestyle="--", alpha=0.4)
        ax_tx.axhline(0, color="gray", lw=0.5)
        ax_tx.axvline(0, color="gray", lw=0.5)

        # ── Receiver output (bottom row) ──
        for idx in range(model.N_users[c]):
            color= cmap(idx)
            ax_rx.scatter(rm[:, idx].real, rm[:, idx].imag, label=f"User {idx}", zorder=3,color=color)
            for sym in range(N_symbols):
                ax_rx.annotate(
                    bits_labels[sym],
                    (rm[sym, idx].real.item(), rm[sym, idx].imag.item()),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, fontweight="bold"
                )
        ax_rx.set_title(f"Channel {c} — Receiver output")
        ax_rx.set_xlabel("Real")
        ax_rx.set_ylabel("Imag")
        ax_rx.legend()
        ax_rx.grid(True, linestyle="--", alpha=0.4)
        ax_rx.axhline(0, color="gray", lw=0.5)
        ax_rx.axvline(0, color="gray", lw=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(path, "symbols_Tx_Rx.png"))
    plt.clf()


def _fmt(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{s:.2f}"
