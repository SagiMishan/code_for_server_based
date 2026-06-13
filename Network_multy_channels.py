import os
import numpy as np
import torch
import torch.nn as nn
from torch import tensor

from Network_single_channel import Network_single_channel


def print_grad_sum_weight_decay(model, skip_list=["User_w", "User_b"]):
    sum = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.split(".")[-1] in skip_list:
            print(name + "  grad : {:.5f} , weight : {:.5f}".format(param.grad.norm(2).item(), param.norm(2).item()))
            sum += param.grad.norm(2).item()
    print(sum)


def dB2lin(x):
    if not torch.is_tensor(x):
        x = tensor(x)
    return torch.pow(10, x / 10)


class Network_multy_channel(nn.Module):
    def __init__(self, N_users, N_channels, N_relays, connectaionMatrix, MatcgRR, MatcgSR, MatcgRU, MatcgTU,
                 modCode_order, N_rx, N_tx, demod_type="simple"):
        super(Network_multy_channel, self).__init__()
        self.connectaionMatrix = connectaionMatrix
        self.N_relays = N_relays
        self.N_users = N_users
        self.N_channels = N_channels
        self.N_rx = N_rx
        self.N_tx = N_tx

        self.modCode_order = modCode_order

        self.demod_type = demod_type

        self.MatcgRR = MatcgRR
        self.MatcgSR = MatcgSR
        self.MatcgRU = MatcgRU
        self.MatcgTU = MatcgTU

        useless_per_channel = self.remove_globally_useless_relays()

        self.sub_networks = nn.ModuleList([Network_single_channel(connectaionMatrix=self.connectaionMatrix[idx],
                                                                  N_relays=N_relays,
                                                                  MatcgSR=self.MatcgSR[idx].T,
                                                                  MatcgRR=self.MatcgRR[idx],
                                                                  MatcgRU=self.MatcgRU[idx],
                                                                  MatcgTU=self.MatcgTU[idx],
                                                                  modCode_order=self.modCode_order[idx],
                                                                  N_rx=self.N_rx,
                                                                  N_tx = self.N_tx,
                                                                  demod_type=self.demod_type,
                                                                  useless_relays = useless_per_channel[idx]) for idx in
                                           range(self.N_channels)])
        self.N_layers = [self.sub_networks[c].N_layers for c in range(self.N_channels)]
        self.P = [torch.ones((self.N_relays, 1)) for c in range(self.N_channels)]

    def find_useless_single_channel(self, conn, cgRU):
        N_relays = self.N_relays  # exclude last row/col (TX)

        # Work only on the relay-to-relay submatrix (exclude TX row and col)
        conn_relays = conn[:N_relays, :N_relays]  # shape (N_relays, N_relays)
        cgRU_relays = cgRU[:N_relays]  # shape (N_relays, N_users*N_rx)

        # Step 1: relays that directly connect to a receiver
        reachable = cgRU_relays.abs().sum(dim=1) > 0  # (N_relays,)

        # Step 2: backward propagation through relay-only connection matrix
        changed = True
        while changed:
            can_reach = (conn_relays * reachable.unsqueeze(0)).sum(dim=1) > 0
            new_reachable = reachable | can_reach
            changed = not torch.equal(new_reachable, reachable)
            reachable = new_reachable

        return set(torch.where(~reachable)[0].tolist())

    def remove_globally_useless_relays(self):
        """
        Remove relays that are useless in ALL channels simultaneously.
        Works purely on raw matrices, before sub-networks are built.
        A relay is useless in a channel if it cannot reach any receiver,
        directly or through other relays.
        connectionMatrix[c] shape: (N_relays, N_relays)
        MatcgRU[c]          shape: (N_relays, N_users*N_rx)
        """

        # Find useless relays per channel
        useless_per_channel = [
            self.find_useless_single_channel(self.connectaionMatrix[c], self.MatcgRU[c])
            for c in range(self.N_channels)
        ]

        # Remove only relays that are useless in ALL channels
        globally_useless = set.intersection(*useless_per_channel)
        globally_useful = sorted(set(range(self.N_relays)) - globally_useless)
        useful_mask = torch.tensor(globally_useful)
        for c in range(self.N_channels):
            print(f"for channel {c} the useless relays are : {sorted(useless_per_channel[c])}")
        print(f"Removing {len(globally_useless)} globally useless relays: {sorted(globally_useless)}")
        print(f"Keeping {len(globally_useful)} relays.")

        # # Prune all matrices
        # new_connectaionMatrix = [
        #     torch.cat([
        #         torch.cat([self.connectaionMatrix[c][:self.N_relays, :self.N_relays][useful_mask][:, useful_mask],
        #                    # relay block
        #                    self.connectaionMatrix[c][:self.N_relays, -1:][useful_mask]], dim=1),  # TX col
        #         self.connectaionMatrix[c][-1:, :][:, torch.cat([useful_mask, torch.tensor([self.N_relays])])]  # TX row
        #     ], dim=0)
        #     for c in range(self.N_channels)
        # ]
        # new_MatcgRU = [self.MatcgRU[c][useful_mask, :] for c in range(self.N_channels)]
        # new_MatcgSR = [self.MatcgSR[c][:, useful_mask] for c in range(self.N_channels)]
        # new_MatcgRR = [self.MatcgRR[c][:, useful_mask][useful_mask, :] for c in range(self.N_channels)]
        #
        # # Update self variables
        # self.N_relays = len(globally_useful)
        # self.MatcgRU = new_MatcgRU
        # self.MatcgSR = new_MatcgSR
        # self.MatcgRR = new_MatcgRR
        # self.connectaionMatrix = new_connectaionMatrix
        return useless_per_channel

    def train_single_channel(self, num_itr, model_idx, loss_fn, optimizer, device, stage3=False, batch=100, SNR=1e-3):
        self.sub_networks[model_idx].SNR = dB2lin(SNR)
        self.sub_networks[model_idx].train()
        optimizer.zero_grad()
        BER = np.zeros((int(num_itr / 100), 1))
        runnig_loss = np.zeros((int(num_itr / 100), 1))
        if stage3:
            self.set_weights()
        for itr in range(num_itr):
            signal ,bits = self.sub_networks[model_idx].modulator(batch)
            # Compute prediction error
            rm = self.sub_networks[model_idx](signal ,bits)
            pred = self.sub_networks[model_idx].demodulator(rm)

            loss = loss_fn(torch.reshape(pred, (-1,)), torch.reshape(bits.to(torch.int32) * 2 - 1, (-1,)))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            # if stage3:
            #     self.set_weights()
            if itr % 100 == 0:
                loss = loss.item()
                runnig_loss[int(itr / 100)] = loss
                worst_BER, avrage_BER, best_BER = self.sub_networks[model_idx].BER(bits=bits, pred=pred)
                BER[int(itr / 100)] = worst_BER
                print("channel {} -- loss: {:.5f} , BER : {:.5f}".format(model_idx, loss, BER[int(itr / 100)][0]))
        return BER, runnig_loss

    def test_single_channel(self, model_idx, loss_fn, batch=100, SNR=torch.tensor(-5), device=None, ToPrint=True):
        self.sub_networks[model_idx].eval()
        test_loss, BER = torch.zeros(SNR.shape), torch.zeros(SNR.shape)
        with torch.no_grad():
            for snr_idx, snr in enumerate(SNR):
                self.sub_networks[model_idx].SNR = dB2lin(snr)
                signal, bits = self.sub_networks[model_idx].modulator(batch)
                # Compute prediction error
                rm = self.sub_networks[model_idx](signal,bits)
                pred = self.sub_networks[model_idx].demodulator(rm)
                # print(f"{signal.shape=}, {bits.shape=}, {rm.shape=}, {pred.shape=}")
                if loss_fn != None:
                    test_loss += loss_fn(pred, bits * 2 - 1).item()
                worst_BER, avrage_BER, best_BER = self.sub_networks[model_idx].BER(bits=bits, pred=pred)
                BER[snr_idx] = worst_BER
                test_loss[snr_idx] /= bits.shape[0]

        if ToPrint:
            for snr_idx, snr in enumerate(SNR):
                print("channel {} -- for SNR = {} , loss: {:.5f} , BER : {:.5f}".format(model_idx, snr,
                                                                                        test_loss[snr_idx],
                                                                                        BER[snr_idx]))
        return BER, test_loss

    def update_v(self):
        for idx in range(self.N_channels):
            self.sub_networks[idx].update_v()

    def learn_score(self):
        self.update_v()
        relay_v_sum = torch.zeros((self.N_relays, 1))
        relay_v_max = -torch.ones((self.N_relays, 1))
        score = 0
        C = self.N_channels
        for c in range(self.N_channels):
            relay_v_sum = relay_v_sum + self.sub_networks[c].V
            relay_v_max = torch.maximum(relay_v_max, self.sub_networks[c].V)
        relay_v_sum = torch.maximum(relay_v_sum, tensor(1e-6))
        relay_v_max = torch.maximum(relay_v_max, tensor(1e-6))
        score = score + torch.sum(C / (C - 1) * relay_v_max / relay_v_sum - 1 / (C - 1))
        return score / self.N_relays

    def sum_v(self):
        sum_v = tensor(0)
        for idx in range(self.N_channels):
            self.sub_networks[idx].update_v()
            sum_v = sum_v + self.sub_networks[idx].sum_v()
        return sum_v

    def culc_p(self):
        self.update_v()
        # Stack all V tensors into a single [N_channels, N_relays, 1] tensor —
        # sum and assignment become single tensor ops, no Python loops.
        V_stack = torch.stack([self.sub_networks[c].V for c in range(self.N_channels)], dim=0)  # [C, N, 1]
        relay_v_sum = V_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)  # [1, N, 1]
        P_stack = V_stack / relay_v_sum  # [C, N, 1]
        for c in range(self.N_channels):
            self.P[c] = P_stack[c]

    def drop_weights(self, z0=0.1):
        p_r = torch.stack([self.P[c].detach().squeeze() for c in range(self.N_channels)])

        bias_probability = (torch.rand(self.N_relays) <= z0).unsqueeze(0).expand_as(p_r)

        # True  → zero out,  False → keep
        to_zero_out = p_r <= torch.rand(p_r.shape)
        # Always keep the best channel for every relay
        to_zero_out[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = False
        # Only zero out where both the random draw AND the bias gate say so
        # (to_zero_out AND bias_probability) → zero;  negate → keep mask
        keep_mask = ~(to_zero_out & bias_probability)  # [C, N]

        n_dropped = (~keep_mask).sum().item()
        if n_dropped > 0:
            for c in range(self.N_channels):
                mask = keep_mask[c].unsqueeze(1)  # [N, 1]
                self.sub_networks[c].w.data.mul_(mask)
                self.sub_networks[c].b.data.mul_(mask)

        return n_dropped

    def set_weights(self):
        # Build p_r in one shot
        p_r = torch.stack([self.P[c].detach().squeeze() for c in range(self.N_channels)])  # [C, N]

        winner = torch.zeros_like(p_r)
        winner[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = 1.0

        for c in range(self.N_channels):
            mask = winner[c].unsqueeze(1).bool()  # [N, 1]

            self.sub_networks[c].w.data.mul_(mask)
            self.sub_networks[c].b.data.mul_(mask)

            # FIX: remove any previously registered hooks before adding new ones,
            # otherwise hooks stack up and multiply the gradient by mask^N.
            if hasattr(self.sub_networks[c], '_mask_hook_handles'):
                for handle in self.sub_networks[c]._mask_hook_handles:
                    handle.remove()

            h_w = self.sub_networks[c].w.register_hook(self.mask_hook(mask))
            h_b = self.sub_networks[c].b.register_hook(self.mask_hook(mask))
            self.sub_networks[c]._mask_hook_handles = [h_w, h_b]

    def mask_hook(self, mask):
        def hook(grad):
            return grad * mask.float()

        return hook

    def selective_gradient(self, tensor, mask):

        if tensor.shape != mask.shape:
            raise ValueError(f"Shape mismatch: tensor shape {tensor.shape}, mask shape {mask.shape}")

        with torch.no_grad():
            # Clone tensor to preserve original
            result = tensor.clone()
            # Detach and zero out the parts where mask is False
            result[~mask] = result[~mask].detach()
        return result

    def mask_grad(self):
        for r in range(self.N_relays):
            p_r = torch.zeros((self.N_channels, 1))
            for c in range(self.N_channels):
                p_r[c, :] = self.P[c][r]

            not_to_zero_or_to_zero = torch.zeros(p_r.shape)
            not_to_zero_or_to_zero[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = 1
            for c in range(self.N_channels):
                mask = torch.unsqueeze(not_to_zero_or_to_zero[c, :], 1).to(torch.bool)
                if self.sub_networks[c].w[r].grad is not None:
                    self.sub_networks[c].w[r].grad *= mask
                if self.sub_networks[c].b[r].grad is not None:
                    self.sub_networks[c].b[r].grad *= mask

    def save(self, path, Name):
        for c in range(self.N_channels):
            self.sub_networks[c].save_python(path, Name + f"c{c}")
            self.sub_networks[c].save_matlab(path, Name + f"c{c}")

    def load(self, path, Name):
        for c in range(self.N_channels):
            self.sub_networks[c].load(path, Name + f"c{c}")



def load_model(path, stage = None):
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
    if not stage is None:
        model.load(path, f'stage_{stage}')
    return model