import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch import matmul, sqrt, tensor
from torch import t as transpose
from scipy.io import savemat
import scipy.io


def save_var_with_name(var, name, prefix=''):
    """
    Save a variable (list, dict, or scalar/array) to .mat files.
    """
    def _to_saveable(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return v

    if isinstance(var, list):
        for idx, elem in enumerate(var):
            new_name = f"{name}_{idx}"
            save_var_with_name(elem, new_name, prefix)
    elif isinstance(var, dict):
        filename = os.path.join(prefix, f"{name}.mat")
        mdict = {str(k): _to_saveable(v) for k, v in var.items()}
        scipy.io.savemat(filename, mdict)
    else:
        filename = os.path.join(prefix, f"{name}.mat")
        scipy.io.savemat(filename, {name: _to_saveable(var)})


def dB2lin(x):
    return torch.pow(10, torch.tensor(x) / 10)

def rapp(A, A0=1, p=3):
    A_mag = torch.abs(A)
    mag = A_mag / (1 + ((A_mag / A0) ** 2) ** p) ** (1 / (2 * p))
    return mag * A / (A_mag + 1e-9)
# Define the full neural network
class Network_single_channel(nn.Module):
    def __init__(self, connectaionMatrix, N_relays, MatcgRR, MatcgSR, MatcgRU, MatcgTU, modCode_order, N_rx,N_tx,useless_relays,
                 demod_type="simple"):
        """
        gain_matrices: List of gain matrices, one for each layer
        """
        super(Network_single_channel, self).__init__()
        self._device = MatcgRR.device
        self.connectaionMatrix = connectaionMatrix
        self.HierarchyLayers = self.NetworkHierarchyLayers(connectaionMatrix)
        self.N_layers = len(self.HierarchyLayers)
        # print(self.HierarchyLayers)

        self.modCode_order = modCode_order
        self.N_rx = N_rx
        self.N_tx = N_tx
        self.register_buffer('modulation', tensor(
            loadmat("./modulation/QAM_{}.mat".format(self.modCode_order))["modulation"][0], dtype=torch.complex64))
        self.demod_type = demod_type

        self.N_users = int( MatcgRU.shape[1] / self.N_rx)
        self.N_relays = N_relays

        self.cgRR = self.ChannelGainRelay2Relay(MatcgRR)
        self.cgSR = self.ChannelGainSource2Relay(MatcgSR)
        self.cgRU = self.ChannelGainRelay2User(MatcgRU)
        self.MatcgTU = MatcgTU

        self.transmitNN = TransmitorNN(N_users = self.N_users,N_tx=N_tx)

        w = self.randomComplexNormal((self.N_relays, 1), sigma=0.1)
        w[list(useless_relays)] = torch.tensor(0,dtype=torch.complex64)
        self.w = nn.Parameter(w)

        b = self.randomComplexNormal((self.N_relays, 1), sigma=0.1)
        b[list(useless_relays)] = torch.tensor(0,dtype=torch.complex64)
        self.b = nn.Parameter(b)

        self.V = torch.ones((self.N_relays, 1))
        self.BN = [ComplexBatchNorm1d(num_of_dim=self.HierarchyLayers[i].size(0)) for i in range(1, self.N_layers)]

        self.User_w = nn.Parameter(self.randomComplexNormal((self.N_users , 1), sigma=0.1))
        self.User_b = nn.Parameter(self.randomComplexNormal((self.N_users , 1), sigma=0.1))
        self.User_BN = ComplexBatchNorm1d(num_of_dim=self.N_users * self.N_rx)


        self.reciverNN = nn.ModuleList(ReciverNN(N_rx=N_rx) for _ in range(self.N_users))


        self.SNR = 1  # in linear form

        self._init_args = {
            "N_relays": N_relays,
            "connectaionMatrix": connectaionMatrix,
            "MatcgRR": MatcgRR,
            "MatcgSR": MatcgSR,
            "MatcgRU": MatcgRU,
            "modCode_order": modCode_order,
            "demod_type": demod_type,

        }
        self.BN_args = {
            "BN": [bn._get_init_arges() for bn in self.BN],
            "BN_user": self.User_BN._get_init_arges()
        }

        self._numBits_ = int(self.modCode_order - 1).bit_length()
        self.register_buffer('_bits_mask_', 2 ** torch.arange(self._numBits_ - 1, -1, -1))

    def _apply(self, fn):
        super()._apply(fn)
        self._device = self.w.device
        self.HierarchyLayers = [fn(t) for t in self.HierarchyLayers]
        self.cgRR = [[fn(t) for t in gain] for gain in self.cgRR]
        self.cgSR = [fn(t) for t in self.cgSR]
        self.cgRU = [fn(t) for t in self.cgRU]
        self.MatcgTU = fn(self.MatcgTU)
        return self

    def forward(self, s,bits):
        batch_size = bits.shape[1]
        if self.demod_type == "complex":

            x = self.transmitNN(bits.to(torch.complex64).T).T
        else:
            x = torch.unsqueeze(s,dim=0)
        r = torch.zeros((self.N_users * self.N_rx, batch_size), dtype=torch.complex64, device=self._device)
        layer_output = []
        for currentIndex, currentLayer in enumerate(self.HierarchyLayers[1:]):
            y = torch.zeros((currentLayer.shape[0], batch_size), dtype=torch.complex64, device=self._device)
            y = y + self.cgSR[currentIndex].T @ x

            for prev_index, prevLayer in enumerate(self.HierarchyLayers[1:currentIndex + 1]):
                y = y + self.cgRR[currentIndex - 1][prev_index] @ layer_output[prev_index]
            y = y + self.randomComplexNormal(y.shape, 1 / self.SNR)
            y = self.BN[currentIndex](y, self.training)

            oil = rapp(y * self.w[currentLayer] + self.b[currentLayer]).type(torch.complex64)

            layer_output.append(oil)

        for currentIndex, currentLayer in enumerate(self.HierarchyLayers[1:]):
            r = r + (transpose(self.cgRU[currentIndex]) @ layer_output[currentIndex]).squeeze()
        r = r + self.MatcgTU.T @ x + self.randomComplexNormal(r.shape, 1 / self.SNR)
        # r = self.User_BN(r, self.training).reshape((self.N_users, self.N_rx,batch_size)).sum(dim=1)
        # r = r * self.User_w + self.User_b
        r = self.User_BN(r, self.training).reshape((self.N_users, self.N_rx,batch_size))
        if self.demod_type == "complex":
            r = torch.cat( [self.reciverNN[m](r[m,:,:].T).T for m in range(self.N_users)],dim=0)

        return r

    def NetworkHierarchyLayers(self, connectionMatrix):
        matrix = connectionMatrix
        Layers = []
        indexes = torch.arange(matrix.shape[0])
        source_index = max(indexes)
        while not matrix.shape[0] <= 1:
            # check the number of connection to the next layer
            numOfConnections = torch.sum(matrix, axis=1)
            # if it is zero it means you are a leaf and that is a layer
            leafs = torch.where(numOfConnections == 0)[0]
            if source_index in indexes[leafs]:
                print("source have source :(" + "-" * 20)
                continue

            # save the leafs index
            Layers.append(indexes[leafs])
            # remove leaves from the matrix
            mask = torch.ones(matrix.shape[0], dtype=torch.bool)
            mask[leafs] = False
            matrix = matrix[mask][:, mask]

            # Update the indexes
            indexes = indexes[mask]
        Layers.append(indexes)
        # Layers.append([source_index])
        Layers.reverse()

        # for idx, layer in enumerate(Layers):
        #     plt.scatter(idx * torch.ones(layer.shape), layer)
        # plt.show()

        return Layers

    def ChannelGainRelay2Relay(self, MatcgRR):
        cgRR = []
        for currentIndex, currentLayer in enumerate(self.HierarchyLayers[2:]):
            gain = []
            for prev_index, prevLayer in enumerate(self.HierarchyLayers[1:(currentIndex + 2)]):
                gain.append(MatcgRR[np.ix_(prevLayer, currentLayer)].T)
            cgRR.append(gain)
        return cgRR

    def ChannelGainRelay2User(self, MatcgRU):
        cgRU = []
        for currentIndex, currentLayer in enumerate(self.HierarchyLayers[1:]):
            cgRU.append(MatcgRU[currentLayer, :].type(torch.complex64))
        return cgRU

    def ChannelGainSource2Relay(self, MatSR):
        cgSR = []
        for currentIndex, currentLayer in enumerate(self.HierarchyLayers[1:]):
            cgSR.append(MatSR[:,currentLayer])
        return cgSR

    def randomComplexNormal(self, shape, sigma=1.0, mu=0.0):
        if torch.is_tensor(sigma):
            sigma = sigma.to(self._device)
        return sigma * torch.randn(size=shape, dtype=torch.complex64, device=self._device) + mu

    def modulator(self, batch_size):
        # symbols = torch.randint(0, self.modCode_order, (batch_size,))
        # bits = self.symbols_to_bits(symbols)

        bits = torch.rand(self.N_users, batch_size, device=self._device) > 0.5
        symbols = torch.sum(torch.unsqueeze(self._bits_mask_, dim=1) * bits, dim=0)
        rm = self.modulation[symbols.to(torch.int32)]
        return rm, bits

    def symbols_to_bits(self, symbols):
        """
        Convert integer modulation symbols to their binary bit representation.

        Each input symbol is assumed to be an integer in the range
        ``[0, self.modCode_order - 1]`` and is expanded into a fixed-length
        binary vector with one row per symbol. The number of bits per symbol
        is determined from ``self.modCode_order`` as
        ``num_bits = ceil(log2(self.modCode_order))``.[web:109][web:118]

        The bits are ordered from most-significant bit (MSB) to least-significant
        bit (LSB) and concatenated into a 1D tensor:
        ``[sym0_bit0, sym0_bit1, ..., sym0_bit{num_bits-1},
           sym1_bit0, ..., sym1_bit{num_bits-1}, ...]``.

        Args:
            symbols (Tensor): 1D integer tensor of shape ``(N,)`` containing
                modulation symbol indices on any device supported by PyTorch.[web:101]

        Returns:
            Tensor: 1D integer tensor of shape ``(N * numBits,)`` containing
            the concatenated bits (values 0 or 1) corresponding to the input
            symbols.

        """

        bits = (symbols.unsqueeze(-1) & self._bits_mask_) > 0
        bits = bits.int()
        return torch.reshape(bits, (-1,))


    def demodulator(self, rm):
        return torch.real(rm)

    def sum_v(self):
        return sum(self.V)

    def update_v(self):
        lambda_w = 1
        lambda_b = 1
        # clamp prevents norm gradient exploding to NaN at exactly zero
        w_norm = self.w.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
        b_norm = self.b.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
        self.V = torch.clamp(1 - torch.exp(-lambda_w * w_norm - lambda_b * b_norm),max=1-1e-8,min=1e-8)

    def BER(self, bits, pred):
        pred = pred > 0
        corrct_or_not = pred != bits
        avrage_BER = corrct_or_not.sum() / (corrct_or_not.shape[0]*corrct_or_not.shape[1])
        BER_per_user = corrct_or_not.sum(dim=1) / corrct_or_not.shape[1]
        best_BER = min(BER_per_user)
        worst_BER = max(BER_per_user)
        return worst_BER, avrage_BER, best_BER

    def update_init_arges(self):

        self.BN_args = {
            "BN": [bn._get_init_arges() for bn in self.BN],
            "BN_user": self.User_BN._get_init_arges()
        }

    def save_python(self, path, Name):
        self.update_init_arges()
        torch.save(
            {
                "init_args": self._init_args,
                "BN_args": self.BN_args,
                "state_dict": self.state_dict()
            },
            os.path.join(path, "models", "python", Name)
        )

    def save_dict_to_npz(self, tensor_dict, save_path):
        # If tensors, convert to numpy arrays
        converted = {k: v.detach().cpu().numpy() if hasattr(v, 'detach') else v for k, v in tensor_dict.items()}
        savemat(save_path, converted)

    def save_matlab(self, path, Name):
        self.update_init_arges()
        save_var_with_name(self.BN_args, Name + "BN_args", os.path.join(path, "models", "matlab"))
        save_var_with_name(self.state_dict(), Name + "state_dict", os.path.join(path, "models", "matlab"))

        # self.save_dict_to_npz(self._init_args, os.path.join(path, "models", "matlab", Name + "_init_arges.mat"))
        # self.save_dict_to_npz(self.BN_args, os.path.join(path, "models", "matlab", Name + "BN_args.mat"))
        # self.save_dict_to_npz(self.state_dict(), os.path.join(path, "models", "matlab", Name + "state_dict.mat"))

    def load(self, path, Name, device=None):
        checkpoint = torch.load(os.path.join(path, "models", "python", Name), map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        dev = self.w.device
        BN_arc = checkpoint["BN_args"]["BN"]

        for idx, bn in enumerate(self.BN):
            self.BN[idx].mean = torch.tensor(BN_arc[idx]["mean"], dtype=torch.complex64).to(dev)
            self.BN[idx].var = torch.tensor(BN_arc[idx]["var"], dtype=torch.complex64).to(dev)
        BN_user_arc = checkpoint["BN_args"]["BN_user"]
        self.User_BN.mean = torch.tensor(BN_user_arc["mean"], dtype=torch.complex64).to(dev)
        self.User_BN.var = torch.tensor(BN_user_arc["var"], dtype=torch.complex64).to(dev)


class ComplexBatchNorm1d:
    def __init__(self, num_of_dim, init_mean=0, init_var=1, momentum=0.1, eps=1e-5):
        self.mean = init_mean * torch.ones((num_of_dim, 1), dtype=torch.complex64, requires_grad=False)
        self.var = init_var * torch.ones((num_of_dim, 1), dtype=torch.complex64, requires_grad=False)
        self.momentum = momentum
        self.eps = eps

    def __call__(self, y, state):
        self.mean = self.mean.to(y.device)
        self.var = self.var.to(y.device)
        if state:  # in train
            with torch.no_grad():
                batch_mean = torch.mean(y, dim=1, keepdim=True)
                batch_var = torch.sqrt(torch.var(y, dim=1, keepdim=True).real.clamp(min=1e-6)).to(torch.complex64)
                m = self.momentum
                self.mean = (1 - m) * self.mean + m * batch_mean
                self.var = (1 - m) * self.var + m * batch_var
        return (y - self.mean) / (self.var + self.eps)

    def _get_init_arges(self):
        return {"mean": self.mean.detach().cpu().numpy(), "var": self.var.detach().cpu().numpy()}

class ReciverNN(nn.Module):
    def __init__(self, N_rx, K=1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2*N_rx, 4*N_rx*K, 2*K]

        dims = [N_rx] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1],dtype=torch.complex64))
        layers.append(nn.Linear(dims[-1], K,dtype=torch.complex64))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransmitorNN(nn.Module):
    def __init__(self, N_users, N_tx, hidden_dims=None, use_bn=True, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2*N_users, 4*N_users*N_tx, 2*N_tx]

        dims = [N_users] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1],dtype=torch.complex64))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-1], N_tx,dtype=torch.complex64))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # return self.complex_tanh( x)
        return rapp(x)
    # def complex_tanh(self,x):
    #     magnitude = x.abs().clamp(min=1e-8)
    #     scale = torch.tanh(magnitude) / magnitude
    #     return x * scale

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
