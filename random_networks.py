import os
import sys
from tqdm import tqdm
original_stdout = sys.stdout

import matplotlib.pyplot as plt

from Network_multy_channels import Network_multy_channel
from Train_funcatios_clude import *
from SmallFunctions import *

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

Name_of_model = "comper_results_single_TX_RX_anntenas"
N_networks = 100
SNR_basic_trainning = 50
max_iteration = 10
SNR_step = 1
BER_th = 1e-3

path = ".\\" + Name_of_model + "\\"
MatcgRR = torch.load(path + "\\data\\MatcgRR.pt", weights_only=True)
MatcgSR = torch.load(path + "\\data\\MatcgSR.pt", weights_only=True)
cgRU = torch.load(path + "\\data\\cgRU.pt", weights_only=True)
cgTU = torch.load(path + "\\data\\cgTU.pt", weights_only=True)
connectaionMatrix = torch.load(path + "\\data\\connectaionMatrix.pt", weights_only=True)
N_users = torch.load(path + "\\data\\N_users.pt", weights_only=True)
N_channels = torch.load(path + "\\data\\N_channels.pt", weights_only=True)
N_relays = torch.load(path + "\\data\\N_relays.pt", weights_only=True)
demod_type = torch.load(path + "\\data\\demod_type.pt", weights_only=True)
posT = torch.load(path + "\\data\\posT.pt", weights_only=False)
posR = torch.load(path + "\\data\\posR.pt", weights_only=False)
posU = torch.load(path + "\\data\\posU.pt", weights_only=False)
N_tx = torch.load(path + "\\data\\N_tx.pt", weights_only=True)
N_rx = torch.load(path + "\\data\\N_rx.pt", weights_only=True)

d_spacing  = torch.load(path + "\\data\\d_spacing.pt",  weights_only=True)
max_snr_train = torch.load(path + "\\data\\max_snr_train_stage_1", weights_only=True)
modCode_order = [2 ** N for N in N_users]


model = Network_multy_channel(N_users=N_users,
                              N_relays=N_relays,
                              N_channels=N_channels,
                              connectaionMatrix=connectaionMatrix,
                              MatcgSR=MatcgSR,
                              MatcgRR=MatcgRR,
                              MatcgRU=cgRU,
                              MatcgTU = cgTU,
                              modCode_order=modCode_order,
                              demod_type=demod_type,
                              N_rx=N_rx,
                              N_tx=N_tx
                              ).to(device)

if not os.path.exists(".\\" + Name_of_model + "\\random_selections\\"):
    os.makedirs(".\\" + Name_of_model + "\\random_selections\\")
main_path = ".\\" + Name_of_model + "\\"
for t in tqdm(range(N_networks)):


    model.load(main_path, "stage_1")


    for c in range(model.N_channels):
        model.P[c] = torch.rand(model.P[c].shape)

    model.set_weights()

    p_r = torch.stack([model.P[c].detach().squeeze() for c in range(model.N_channels)])  # [C, N]

    winner = torch.zeros_like(p_r)
    winner[torch.argmax(p_r, dim=0), torch.arange(p_r.shape[1])] = 1.0

    name_of_model_random = winner[0,:].to(dtype=torch.int)
    # print(name_of_model_random)

    binary_str = ''.join(map(str, name_of_model_random.flatten().tolist()))

    # Pad the binary string to ensure it is divisible by 4
    padded_binary_str = binary_str.zfill((len(binary_str) + 3) // 4 * 4)
    # Convert each group of 4 bits to a HEX digit
    name_of_model_random = ''.join(f'{int(padded_binary_str[i:i + 4], 2):X}'
                            for i in range(0, len(padded_binary_str), 4))
    print(name_of_model_random)

    path = main_path +  "random_selections\\"+name_of_model_random + "\\"
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
                SNR_max=max_snr_train + 10,
                BER_th=BER_th,
                device=device,
                SNR_step=SNR_step,
                max_iteration=max_iteration,
                path=path,
                SNR_val= 0)
        SNR = torch.linspace(max_snr_train-10, max_snr_train+20, 31)
        worst_BER, best_BER = plotSNRvsBER(model, num_itr=1,
                                       batch=10**5,
                                       SNR=SNR,stage=name_of_model_random)
        plot_architecture(path=main_path,
                          Name_of_model= "random_selections\\"+name_of_model_random + "\\", stage="stage_3")
        plt.close()
    finally:
        print("done")
        # sys.stdout.close()
        # sys.stdout = original_stdout

    torch.save(worst_BER, path + "outputs\\worst_BER.pt")
    torch.save(best_BER, path + "outputs\\best_BER.pt")
    torch.save(SNR, path + "outputs\\SNR.pt")
    plt.close()
