import os
import sys
import itertools

from torch._dynamo.variables import torch
from tqdm import tqdm
import math
import random
original_stdout = sys.stdout

import matplotlib.pyplot as plt

from code_Networks.Network_multy_channels import Network_multy_channel,load_model
from code_Networks.Train_funcatios_clude import *
from code_Networks.SmallFunctions import plotSNRvsBER,plot_architecture,save_var_with_name


def generate_tensors_with_hamming_distance(tensor, d, x=-1, exclude_indices=None):
    """
    Generate tensors at Hamming distance d from the input tensor.

    Args:
        tensor:          Binary input tensor (1D)
        d:               Hamming distance (number of bits to flip)
        x:               Max number of random samples to return.
                         If x == -1 (or x >= total combinations), return ALL combinations.
        exclude_indices: Set of indices that are locked and cannot be flipped.

    Returns:
        List of tensors, each differing from `tensor` by exactly d flipped bits,
        with no flips at the excluded indices.
    """
    n = tensor.numel()

    # Only allow flipping on non-excluded indices
    if exclude_indices is not None:
        indices = [i for i in range(n) if i not in exclude_indices]
    else:
        indices = list(range(n))

    total = math.comb(len(indices), d)

    def make_tensor(combo):
        new_tensor = tensor.clone()
        for idx in combo:
            new_tensor[idx] = 1 - new_tensor[idx]
        return new_tensor

    if x == -1 or x >= total:
        return [make_tensor(combo) for combo in itertools.combinations(indices, d)]

    seen = set()
    result_tensors = []

    while len(result_tensors) < x:
        combo = tuple(sorted(random.sample(indices, d)))
        if combo not in seen:
            seen.add(combo)
            result_tensors.append(make_tensor(combo))

    return result_tensors# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(123456)
Name_of_model = "model_4_comper_results_single_TX_RX_anntenas"
N_models = 50
max_distance = 10
SNR_basic_trainning = 50
SNR_max = -3
max_iteration = 5
SNR_step = 1
BER_th = 1e-3

main_path = ".\\" + Name_of_model + "\\"



if not os.path.exists(".\\" + Name_of_model + f"\\close_selections_distance_{max_distance}\\"):
    os.makedirs(".\\" + Name_of_model + f"\\close_selections_distance_{max_distance}\\")

model = load_model(path=main_path,stage=3)
model.load( ".\\" + Name_of_model,"stage_3")
model.culc_p()

model.set_weights()

p_r = torch.stack([model.P[c].detach().squeeze() for c in range(model.N_channels)])  # [C, N]

winner = torch.zeros_like(p_r)
winner[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = 1.0

name_of_model_int = winner[0, :].to(dtype=torch.int)
name_of_model_int = name_of_model_int.clone()
print(name_of_model_int)

binary_str = ''.join(map(str, name_of_model_int.flatten().tolist()))

# Pad the binary string to ensure it is divisible by 4
padded_binary_str = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
# Convert each group of 4 bits to a HEX digit
name_of_model_0x = ''.join(f'{int(padded_binary_str[i:i + 4], 2):X}'
                        for i in range(0, len(padded_binary_str), 4))
print(name_of_model_0x)
all_comb= generate_tensors_with_hamming_distance(name_of_model_int,max_distance,N_models)

# exclude_indices = model.remove_globally_useless_relays()
# exclude_indices = set.union(*exclude_indices)
# all_comb = generate_tensors_with_hamming_distance(name_of_model_int,max_distance,N_models,exclude_indices)

for comb in tqdm(all_comb):
    path = main_path
    model.load(main_path, "stage_1")

    model.P[0] = comb
    model.P[1] = (comb-1)**2

    model.set_weights()

    # print(name_of_model_random)

    binary_str = ''.join(map(str, comb.flatten().tolist()))

    # Pad the binary string to ensure it is divisible by 4
    padded_binary_str = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
    # Convert each group of 4 bits to a HEX digit
    name_of_model_random = ''.join(f'{int(padded_binary_str[i:i + 4], 2):X}'
                            for i in range(0, len(padded_binary_str), 4))
    print(name_of_model_random)

    path += f"close_selections_distance_{max_distance}\\"+name_of_model_random + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "\\models")
        os.makedirs(path + "\\models\\python")
        os.makedirs(path + "\\models\\matlab")
        os.makedirs(path + "\\outputs")
        os.makedirs(path + "\\data")

        # sys.stdout = open(os.devnull, 'w')
        try:

            stage_3(model=model,
                    SNR_basic_trainning=SNR_basic_trainning,
                    SNR_max=SNR_max,
                    BER_th=BER_th,
                    device=device,
                    SNR_step=SNR_step,
                    max_iteration=max_iteration,
                    path=path,
                    SNR_val=0)
            # SNR = torch.linspace(0, 15, 16)
            # worst_BER, best_BER = plotSNRvsBER(model, num_itr=1,
            #                                    batch=10 ** 5,
            #                                    SNR=SNR, stage=name_of_model_random)
            plot_architecture(path=main_path,
                              Name_of_model=f"close_selections_distance_{max_distance}\\"+name_of_model_random +"\\", stage="stage_3")
            plt.close()
        finally:
            print("done")
            # sys.stdout.close()
            # sys.stdout = original_stdout
    SNR = torch.linspace(-20, 20, 41)
    worst_BER, best_BER = plotSNRvsBER(model, num_itr=5,
                                       batch=10 ** 5,
                                       SNR=SNR, stage=name_of_model_random)
    torch.save(worst_BER, path + "outputs\\worst_BER.pt")
    torch.save(best_BER, path + "outputs\\best_BER.pt")
    torch.save(SNR, path + "outputs\\SNR.pt")
