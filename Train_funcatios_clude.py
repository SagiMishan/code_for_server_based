import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from SmallFunctions import Single_channel_loss_function, Multy_channel_loss_function, dB2lin


def _add_z0_axis(ax_main, itr, z0_ds, color_z0='tab:orange'):
    """
    Attach a right-hand y-axis showing the z0 schedule.

    itr   : 1-D array of iteration indices (must match the main plot x-axis)
    z0_ds : z0 values sampled at those exact iteration indices
    """
    ax_z0 = ax_main.twinx()
    ax_z0.plot(itr, z0_ds, color=color_z0, linewidth=1.2,
               linestyle=':', alpha=0.75, label='z0 schedule')
    ax_z0.set_ylabel('z0 (bias probability)', color=color_z0, fontsize=9)
    ax_z0.tick_params(axis='y', labelcolor=color_z0)
    ax_z0.set_ylim(-0.05, 1.15)
    return ax_z0


def _combined_legend(ax_main, ax_z0, **legend_kwargs):
    """Merge legend handles from two axes into one box placed outside the plot."""
    h1, l1 = ax_main.get_legend_handles_labels()
    h2, l2 = ax_z0.get_legend_handles_labels()
    defaults = dict(loc='upper left', bbox_to_anchor=(1.12, 1),
                    borderaxespad=0, fontsize=8)
    defaults.update(legend_kwargs)
    ax_main.legend(h1 + h2, l1 + l2, **defaults)


def _save(fig, path, name):
    """Save figure and always close it cleanly."""
    fig.savefig(os.path.join(path, 'outputs', name), bbox_inches='tight', dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting routine – called once per epoch from stage_2
# ─────────────────────────────────────────────────────────────────────────────

def _plot_all(log: dict, z0_full: torch.Tensor, path: str,
              epoch, window: int = 10):
    """
    Produce one PNG per metric, all with the z0 schedule on a twin axis.
    epoch can be an int (for per-epoch plots) or the string 'all' (cumulative).
    """
    tag = 'all' if epoch == 'all' else f'e{epoch:02d}'
    title_suffix = 'all epochs' if epoch == 'all' else f'epoch {epoch}'

    itr = np.array(log['itr_axis'])
    C = len(log['BER_per_ch'])
    ma = np.ones(window) / window

    # z0 sampled at the exact logged iteration indices — correct x-alignment
    z0_ds = z0_full[itr].numpy()

    colors_ch = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ── 1. Loss ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(itr, log['loss'], color='steelblue', alpha=0.4, linewidth=1, label='Loss (raw)')
    if len(log['loss']) >= window:
        ma_loss = np.convolve(log['loss'], ma, mode='valid')
        ax.plot(itr[window - 1:], ma_loss, color='steelblue', linewidth=2,
                label=f'Loss (MA={window})')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Loss');
    ax.set_title(f'Training loss — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_loss.png')

    # ── 2. V score ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(itr, log['v_score'], color='tab:blue', alpha=0.35, linewidth=1, label='V score (raw)')
    if len(log['v_score']) >= window:
        ma_v = np.convolve(log['v_score'], ma, mode='valid')
        ax.plot(itr[window - 1:], ma_v, color='tab:blue', linewidth=2,
                label=f'V score (MA={window})')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Iteration');
    ax.set_ylabel('V score')
    ax.set_title(f'Channel selection score (V) — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_v_score.png')

    # ── 3. Worst BER (all channels combined) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ber_arr = np.array(log['worst_BER'])
    ber_arr = np.where(ber_arr == 0, np.nan, ber_arr)  # mask zeros for log scale
    ax.semilogy(itr, ber_arr, color='crimson', alpha=0.45, linewidth=1, label='Worst BER (raw)')
    valid = ~np.isnan(ber_arr)
    if valid.sum() >= window:
        ma_ber = np.convolve(ber_arr[valid], ma, mode='valid')
        ax.semilogy(itr[valid][window - 1:], ma_ber, color='crimson', linewidth=2,
                    label=f'Worst BER (MA={window})')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('BER (log scale)')
    ax.set_title(f'Worst-case BER across all channels — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_worst_BER.png')

    # ── 4. Per-channel BER ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(C):
        ber_c = np.array(log['BER_per_ch'][c])
        ber_c = np.where(ber_c == 0, np.nan, ber_c)
        col = colors_ch[c % len(colors_ch)]
        ax.semilogy(itr, ber_c, color=col, alpha=0.35, linewidth=1)
        valid = ~np.isnan(ber_c)
        if valid.sum() >= window:
            ma_c = np.convolve(ber_c[valid], ma, mode='valid')
            ax.semilogy(itr[valid][window - 1:], ma_c, color=col, linewidth=2,
                        label=f'Ch {c} BER')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('BER (log scale)')
    ax.set_title(f'Per-channel worst BER — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_BER_per_channel.png')

    # ── 5. N drops ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(itr, log['n_drops'], width=80, color='slategray', alpha=0.6, label='N drops / 100 itr')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Weights dropped')
    ax.set_title(f'Weight drops per 100 iterations — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_n_drops.png')

    # ── 6. Active relays per channel ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(C):
        col = colors_ch[c % len(colors_ch)]
        ax.plot(itr, log['n_relays_per_ch'][c], color=col, linewidth=1.8,
                label=f'Ch {c} active relays')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Active relays (P > 0.9)')
    ax.set_title(f'Active relays per channel — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_active_relays.png')

    # ── 7. Mean |w| per channel ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(C):
        col = colors_ch[c % len(colors_ch)]
        ax.plot(itr, log['w_norm_per_ch'][c], color=col, linewidth=1.8,
                label=f'Ch {c} mean |w|')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Mean |w|')
    ax.set_title(f'Mean relay gain magnitude per channel — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_w_norm.png')

    # ── 8. Mean |b| per channel ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(C):
        col = colors_ch[c % len(colors_ch)]
        ax.plot(itr, log['b_norm_per_ch'][c], color=col, linewidth=1.8,
                label=f'Ch {c} mean |b|')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Mean |b|')
    ax.set_title(f'Mean relay bias magnitude per channel — {title_suffix}')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_b_norm.png')

    # ── 9. P-distribution entropy per channel ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(C):
        col = colors_ch[c % len(colors_ch)]
        ax.plot(itr, log['p_entropy_per_ch'][c], color=col, linewidth=1.8,
                label=f'Ch {c} P entropy')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Shannon entropy (nats)')
    ax.set_title(f'Relay-assignment probability entropy per channel — {title_suffix}\n'
                 '(low → concentrated assignment, high → diffuse)')
    ax_z0 = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, path, f'{tag}_p_entropy.png')

    # ── 10. Summary panel (2×2 grid) ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Stage 2 summary — {title_suffix}', fontsize=13)

    # top-left: V score
    ax = axes[0, 0]
    ax.plot(itr, log['v_score'], alpha=0.35, color='tab:blue', linewidth=1)
    if len(log['v_score']) >= window:
        ax.plot(itr[window - 1:], np.convolve(log['v_score'], ma, mode='valid'),
                color='tab:blue', linewidth=2, label='V score')
    ax.set_ylim(0, 1.05)
    ax.set_title('V score')
    ax.set_xlabel('Iteration')
    ax_z0a = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0a)

    # top-right: worst BER
    ax = axes[0, 1]
    ber_arr2 = np.where(np.array(log['worst_BER']) == 0, np.nan, np.array(log['worst_BER']))
    ax.semilogy(itr, ber_arr2, alpha=0.4, color='crimson', linewidth=1)
    valid2 = ~np.isnan(ber_arr2)
    if valid2.sum() >= window:
        ax.semilogy(itr[valid2][window - 1:],
                    np.convolve(ber_arr2[valid2], ma, mode='valid'),
                    color='crimson', linewidth=2, label='Worst BER')
    ax.set_title('Worst BER');
    ax.set_xlabel('Iteration')
    ax_z0b = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0b)

    # bottom-left: N drops
    ax = axes[1, 0]
    ax.bar(itr, log['n_drops'], width=80, color='slategray', alpha=0.6, label='N drops')
    ax.set_title('Weight drops / 100 itr');
    ax.set_xlabel('Iteration')
    ax_z0c = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0c)

    # bottom-right: active relays per channel
    ax = axes[1, 1]
    for c in range(C):
        ax.plot(itr, log['n_relays_per_ch'][c],
                color=colors_ch[c % len(colors_ch)], linewidth=1.8,
                label=f'Ch {c}')
    ax.set_title('Active relays per channel');
    ax.set_xlabel('Iteration')
    ax_z0d = _add_z0_axis(ax, itr, z0_ds)
    _combined_legend(ax, ax_z0d)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, path, f'{tag}_summary.png')


def train_multy_channel(model, num_itr, loss_fn, optimizer, path, z0,
                        batch=100, SNR=1e-3, max_V=0):
    """
    Run one epoch of multi-channel training (stage 2).

    Returns
    -------
    BER          : np.ndarray [num_itr//100]  worst BER across channels
    runnig_loss  : np.ndarray [num_itr//100]  total loss value
    SCORE        : torch.Tensor [num_itr//100] V score
    runnig_drops : torch.Tensor [num_itr//100] weight drops per window
    log          : dict  – all raw per-step metrics (see _plot_all for keys)
    """
    C = model.N_channels
    N = model.N_relays
    LOG = num_itr // 100  # number of logging steps

    BER = np.zeros(LOG)
    SCORE = torch.zeros(LOG)
    runnig_loss = np.zeros(LOG)
    runnig_drops = torch.zeros(LOG)
    moving_drops = 0

    # Pre-initialise the per-step log dict
    log = dict(
        itr_axis=[],
        loss=[],
        v_score=[],
        n_drops=[],
        worst_BER=[],
        BER_per_ch={c: [] for c in range(C)},
        n_relays_per_ch={c: [] for c in range(C)},
        w_norm_per_ch={c: [] for c in range(C)},
        b_norm_per_ch={c: [] for c in range(C)},
        p_entropy_per_ch={c: [] for c in range(C)},
        bais_probability=[],
    )

    snr_lin = dB2lin(SNR)
    for c in range(C):
        model.sub_networks[c].SNR = snr_lin

    all_pred = [None] * C
    all_bits = [None] * C

    _device = next(model.parameters()).device
    for itr in range(num_itr):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float, device=_device)

        for c in range(C):
            model.sub_networks[c].train()
            signal, bits = model.sub_networks[c].modulator(batch)
            rm = model.sub_networks[c](signal, bits)
            pred = model.sub_networks[c].demodulator(rm)

            all_pred[c] = pred.detach()
            all_bits[c] = bits

            sum_v_c = model.sub_networks[c].V.sum()
            loss = loss + loss_fn(pred, bits * 2 - 1) + 1e-1 * sum_v_c / N

        loss.backward()
        # if nan output this code will help to find it
        # nan_found = False
        # for name, p in model.named_parameters():
        #     if p.grad is None:
        #         continue
        #     if not torch.isfinite(p.grad).all():
        #         n_nan = (~torch.isfinite(p.grad)).sum().item()
        #         n_tot = p.grad.numel()
        #         print(f"[GRAD NaN] {name}  —  {n_nan}/{n_tot} values non-finite  "
        #               f"|grad|_max={p.grad.abs().nan_to_num().max().item():.4f}")
        #         nan_found = True
        #
        # if nan_found:
        #     # print which relay index is the problem
        #     for c in range(model.N_channels):
        #         w = model.sub_networks[c].w
        #         b = model.sub_networks[c].b
        #         if w.grad is not None:
        #             bad = (~torch.isfinite(w.grad)).any(dim=1)
        #             if bad.any():
        #                 print(f"  ch{c} bad relay indices (w.grad): {bad.nonzero().squeeze().tolist()}")
        #         if b.grad is not None:
        #             bad = (~torch.isfinite(b.grad)).any(dim=1)
        #             if bad.any():
        #                 print(f"  ch{c} bad relay indices (b.grad): {bad.nonzero().squeeze().tolist()}")
        #

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #  if nan as an ouput this code will help to find it
        # for name, p in model.named_parameters():
        #     if not torch.isfinite(p).all():
        #         print(f"  [WARNING] non-finite parameter {name} at itr {itr}")
        #     if p.grad is not None and not torch.isfinite(p.grad.norm()).all():
        #         print(f"  [WARNING] non-finite grad in {name}, zeroing")
        #         p.grad.zero_()

        # for name, p in model.named_parameters():
        #     if not torch.isfinite(p).all():
        #         print(f"  [WARNING] non-finite parameter {name} at itr {itr}")
        #     if p.grad is not None and not torch.isfinite(p.grad.norm()).all():
        #         print(f"  [WARNING] non-finite grad in {name}, zeroing")
        #         p.grad.zero_()
        model.update_v()
        model.culc_p()

        current_drop = model.drop_weights(z0=z0[itr])
        moving_drops += current_drop

        # ── Logging every 100 iterations ─────────────────────────────────────
        if itr % 100 == 0:
            li = itr // 100  # log index
            loss_val = loss.item()
            runnig_loss[li] = loss_val

            score = model.learn_score()
            SCORE[li] = score

            runnig_drops[li] = moving_drops if itr == 0 else moving_drops / 100

            # ── per-channel metrics ───────────────────────────────────────────
            total_worst = 0.0
            score_detach = score.detach()
            out_string = "{:<2}/{} score={:.4f}".format(li + 1, LOG, float(score_detach))

            for c in range(C):
                worst_BER, avg_BER, best_BER = model.sub_networks[c].BER(
                    bits=all_bits[c], pred=all_pred[c])
                total_worst = max(total_worst, float(worst_BER))

                # active relays: P > 0.9
                active = int(sum(model.P[c][:] > 0.9)[0])

                # mean magnitude of w and b for this channel
                w_mag = model.sub_networks[c].w.abs().mean().item()
                b_mag = model.sub_networks[c].b.abs().mean().item()

                # Shannon entropy of P distribution for this channel (nats)
                p_c = model.P[c].detach().squeeze().clamp(min=1e-9)
                entropy = float(-(p_c * p_c.log()).sum())

                log['BER_per_ch'][c].append(float(worst_BER))
                log['n_relays_per_ch'][c].append(active)
                log['w_norm_per_ch'][c].append(w_mag)
                log['b_norm_per_ch'][c].append(b_mag)
                log['p_entropy_per_ch'][c].append(entropy)
                log["bais_probability"].append(z0[itr])
                out_string += "  | ch{} BER={:.3f} relays={} |w|={:.3f} |b|={:.3f}".format(
                    c, float(worst_BER), active, w_mag, b_mag)

            BER[li] = total_worst

            log['itr_axis'].append(itr)
            log['loss'].append(loss_val)
            log['v_score'].append(float(score_detach))
            log['n_drops'].append(float(runnig_drops[li]))
            log['worst_BER'].append(total_worst)

            out_string += "  z0={:.4f}  drops={:.1f}".format(
                float(z0[itr]), runnig_drops[li])
            print(out_string)

            moving_drops = 0
            if abs(1 - score) < 1e-2:
                return BER, runnig_loss, SCORE, runnig_drops, log

    return BER, runnig_loss, SCORE, runnig_drops, log


def _fast_learn_score(model):
    """
    Compute learn_score() without calling update_v() again.
    Assumes model.update_v() was already called this iteration.
    Mirrors Network_multy_channel.learn_score() exactly.
    """
    dev = next(model.parameters()).device
    relay_v_sum = torch.zeros((model.N_relays, 1), device=dev)
    relay_v_max = -torch.ones((model.N_relays, 1), device=dev)
    C = model.N_channels
    for c in range(C):
        v = model.sub_networks[c].V
        relay_v_sum = relay_v_sum + v
        relay_v_max = torch.maximum(relay_v_max, v)
    relay_v_sum = relay_v_sum.clamp(min=1e-6)
    relay_v_max = relay_v_max.clamp(min=1e-6)
    score = torch.sum(C / (C - 1) * relay_v_max / relay_v_sum - 1 / (C - 1))
    return (score / model.N_relays).item()


def stage_1(model, basic_training, SNR_basic_trainning, SNR_max, SNR_step, max_iteration, BER_th,
            device, path):
    loss_fn_SC = Single_channel_loss_function
    if basic_training:
        max_snr_train = -100000
        for c in range(model.N_channels):
            SNR = SNR_max
            flag_BER = False
            optimizer = torch.optim.Adam(model.sub_networks[c].parameters(), lr=1e-3)
            epochs = 0
            while True:
                if not flag_BER:
                    print("stage 1 - channel {} -- Epoch {} - SNR: {:.2f}\n"
                          "-------------------------------".format(c, epochs, SNR_basic_trainning))
                    BER, loss = model.train_single_channel(num_itr=1000, model_idx=c,
                                                           loss_fn=loss_fn_SC, optimizer=optimizer,
                                                           device=device, batch=512,
                                                           SNR=SNR_basic_trainning)
                    epochs += 1
                    if np.mean(BER) == 0:
                        epochs = 0
                        flag_BER = True
                        print("changed flag")
                    if epochs >= max_iteration:
                        break
                else:
                    print("stage 1 - channel {} -- Epoch {} - SNR: {:.2f}\n"
                          "-------------------------------".format(c, epochs, SNR))
                    BER, loss = model.train_single_channel(num_itr=1000, model_idx=c,
                                                           loss_fn=loss_fn_SC, optimizer=optimizer,
                                                           device=device, batch=2 ** 7, SNR=SNR)
                    if np.mean(BER) < max(BER_th, 2 ** -7):
                        SNR = SNR - SNR_step
                        epochs = 0
                    if epochs >= max_iteration:
                        break
                    epochs += 1
                # print("testing")
                # model.test_single_channel(model_idx=c, loss_fn=loss_fn_SC, device=device,
                #                           batch=2 ** 7,
                #                           SNR=torch.linspace(SNR - 10, SNR + 10, 21))
            max_snr_train = max(SNR, max_snr_train)
            print("Done! training")

        torch.save(max_snr_train, path + "\\data\\max_snr_train_stage_1")
        model.save(path, "stage_1")
    else:
        model.load(path, "stage_1")
        max_snr_train = torch.load(path + "\\data\\max_snr_train_stage_1", weights_only=True)

    return max_snr_train


def stage_2(model, max_snr_train, epochs, device, path, z0_type, sub_stages,
            z0_init=0.01, z0_end=0.01, B=torch.tensor(10)):
    loss_fn_SC = Single_channel_loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── build z0 schedule ────────────────────────────────────────────────────
    total_itr = epochs * 1000
    if z0_init == z0_end and z0_type == "const":
        print("stage 2 with Const z0 = {}".format(z0_init))
        z0 = z0_init * torch.ones(total_itr + 1)

    elif (not z0_init == z0_end) and z0_type == "linear":
        print("stage 2 with linear s {} E {}".format(z0_init, z0_end))
        z0 = torch.linspace(z0_init, z0_end, total_itr + 1)

    elif z0_type == "exp":
        print("stage 2 with exp - B = {}".format(B))
        no_drop = torch.zeros(int(total_itr * sub_stages[0]) - 1)
        ramp = torch.exp(
            -B * (1 - torch.arange(0, total_itr * sub_stages[1] + 1)
                  / (total_itr * sub_stages[1])))
        always = torch.ones(int(total_itr * sub_stages[2]))
        z0 = torch.cat((no_drop, ramp, always))
    else:
        raise ValueError(f"Unknown z0_type: {z0_type}")

    # ── accumulators across all epochs ───────────────────────────────────────
    ALL_SCORE = []
    ALL_N_DROPS = []
    ALL_LOGS = []  # one log dict per epoch

    max_V = 0
    for t in range(epochs):
        print("\nstage 2 - Epoch {} / {} - SNR: {:.2f}\n{}".format(
            t + 1, epochs, max_snr_train, '-' * 40))

        z0_epoch = z0[t * 1000: (t + 1) * 1000]

        BER, loss_arr, score, drops, log = train_multy_channel(
            model=model,
            num_itr=1000,
            loss_fn=loss_fn_SC,
            optimizer=optimizer,
            batch=2 ** 7,
            SNR=max_snr_train,
            z0=z0_epoch,
            path=path,
            max_V=max_V,
        )

        ALL_SCORE.append(score)
        ALL_N_DROPS.append(drops)
        ALL_LOGS.append(log)

        # ── save raw data for this epoch ──────────────────────────────────────
        data_ep = dict(
            BER=BER,
            loss=loss_arr,
            score=score.detach().numpy(),
            drops=drops.detach().numpy(),
            z0_epoch=z0_epoch.numpy(),
            log=log,
        )
        # torch.save(data_ep,
        #            os.path.join(path, 'data', f'stage2_epoch{t + 1:02d}_raw.pt'))

        # ── per-epoch plots ───────────────────────────────────────────────────
        # log['itr_axis'] are local (0..999); shift to global iteration index
        global_offset = t * 1000
        log_global = dict(log)  # shallow copy
        log_global['itr_axis'] = [i + global_offset for i in log['itr_axis']]

        # _plot_all(log_global, z0, path, epoch=t + 1)

        # ── early-stop check ─────────────────────────────────────────────────
        if abs(model.learn_score() - 1) <= 1e-2:
            print("got a V score = 1 , stop stage 2")
            epochs = t + 1  # update for the cumulative plots below
            break

    # ── finalise model ────────────────────────────────────────────────────────
    model.set_weights()
    model.save(path, "stage_2")

    # ── cumulative plots across all epochs ────────────────────────────────────
    SCORE_all = torch.cat(ALL_SCORE, dim=0).detach().numpy()
    DROPS_all = torch.cat(ALL_N_DROPS, dim=0).detach().numpy()

    # Build a single merged log for all epochs
    merged = dict(
        itr_axis=[],
        loss=[],
        v_score=[],
        n_drops=[],
        worst_BER=[],
        BER_per_ch={c: [] for c in range(model.N_channels)},
        n_relays_per_ch={c: [] for c in range(model.N_channels)},
        w_norm_per_ch={c: [] for c in range(model.N_channels)},
        b_norm_per_ch={c: [] for c in range(model.N_channels)},
        p_entropy_per_ch={c: [] for c in range(model.N_channels)},
        bais_probability=[]
    )
    for t, lg in enumerate(ALL_LOGS):
        offset = t * 1000
        merged['itr_axis'] += [i + offset for i in lg['itr_axis']]
        merged['loss'] += lg['loss']
        merged['v_score'] += lg['v_score']
        merged['n_drops'] += lg['n_drops']
        merged['worst_BER'] += lg['worst_BER']
        merged['bais_probability'] += lg['bais_probability']
        for c in range(model.N_channels):
            merged['BER_per_ch'][c] += lg['BER_per_ch'][c]
            merged['n_relays_per_ch'][c] += lg['n_relays_per_ch'][c]
            merged['w_norm_per_ch'][c] += lg['w_norm_per_ch'][c]
            merged['b_norm_per_ch'][c] += lg['b_norm_per_ch'][c]
            merged['p_entropy_per_ch'][c] += lg['p_entropy_per_ch'][c]

    # Save merged raw data
    torch.save(merged, os.path.join(path, 'data', 'stage2_all_epochs_raw.pt'))

    # Cumulative plots (epoch = 0 → "all" label)
    _plot_all(merged, z0, path, epoch="all")  # 0 → saved as e00_*.png  (= cumulative)

    print("\nDone! stage 2 training")
    print(f"  Plots saved to  {os.path.join(path, 'outputs')}")
    print(f"  Raw data saved to {os.path.join(path, 'data')}")


def stage_3(model, SNR_basic_trainning, SNR_max, SNR_step, max_iteration, BER_th, device, SNR_val,
            path):
    loss_fn_SC = Single_channel_loss_function
    for c in range(model.N_channels):
        SNR = SNR_max
        epochs = 0
        flag_BER = False
        optimizer = torch.optim.Adam(model.sub_networks[c].parameters(), lr=1e-3)
        while True:
            if not flag_BER:
                print("stage 3 - channel {} -- Epoch {} - SNR: {:.2f}\n"
                      "-------------------------------".format(c, epochs, SNR_basic_trainning))
                BER, loss = model.train_single_channel(num_itr=1000, model_idx=c,
                                                       loss_fn=loss_fn_SC, optimizer=optimizer,
                                                       device=device, batch=512,
                                                       SNR=SNR_basic_trainning, stage3=True)
                epochs += 1
                if np.mean(BER) == 0:
                    epochs = 0
                    flag_BER = True
                    print("changed flag")
                if epochs >= max_iteration:
                    break
            else:
                print("stage 3 - channel {} -- Epoch {} - SNR: {:.2f}\n"
                      "-------------------------------".format(c, epochs, SNR))
                BER, loss = model.train_single_channel(num_itr=1000, model_idx=c,
                                                       loss_fn=loss_fn_SC, optimizer=optimizer,
                                                       device=device, batch=2 ** 7, SNR=SNR,
                                                       stage3=True)
                if np.mean(BER) < max(BER_th, 2 ** -7):
                    SNR = SNR - SNR_step
                    epochs = 0
                if epochs >= max_iteration:
                    break
                epochs += 1
            # print("testing")
            # model.test_single_channel(model_idx=c, loss_fn=loss_fn_SC, device=device,
            #                           batch=2 ** 7,
            #                           SNR=torch.linspace(SNR_val - 10, SNR_val + 10, 21))
    model.save(path, "stage_3")
    print("Done! training")
