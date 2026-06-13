import os
import sys
import time
from tqdm import tqdm

from SmallFunctions import _fmt
original_stdout = sys.stdout

import matplotlib.pyplot as plt

from Network_multy_channels import Network_multy_channel, load_model
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
torch.manual_seed(0)
orignal_Name_of_model = "compare_networks"
N_networks = 2
SNR_basic_trainning = 50
max_iteration = 2
SNR_step = 5
BER_th = 1e-3


for idx in range(10):
    Name_of_model = f"{orignal_Name_of_model}_{idx}"

    main_path = os.path.join(".", Name_of_model, "")

    model = load_model(path=main_path,stage=1)
    max_snr_train = torch.load(os.path.join(main_path, "data", "max_snr_train_stage_1"), weights_only=True)
    if not os.path.exists(os.path.join(".", Name_of_model, "random_selections")):
        os.makedirs(os.path.join(".", Name_of_model, "random_selections"))
    for t in tqdm(range(N_networks)):
        start_time_1 = time.time()

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
        print(f"Selected model in {orignal_Name_of_model}-{t}: {name_of_model_random}")

        path = os.path.join(main_path, "random_selections", name_of_model_random, "")
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, "models"))
            os.makedirs(os.path.join(path, "models", "python"))
            os.makedirs(os.path.join(path, "models", "matlab"))
            os.makedirs(os.path.join(path, "outputs"))
            os.makedirs(os.path.join(path, "data"))
        # sys.stdout = open(os.devnull, 'w')
        try:
            stage_3(model=model,
                    SNR_basic_trainning=SNR_basic_trainning,
                    SNR_max=max_snr_train + 3,
                    BER_th=BER_th,
                    device=device,
                    SNR_step=SNR_step,
                    max_iteration=max_iteration,
                    path=path,
                    SNR_val= 0)
            # SNR = torch.linspace(max_snr_train-10, max_snr_train+20, 31)
            # worst_BER, best_BER = plotSNRvsBER(model, num_itr=1,
            #                                batch=10**5,
            #                                SNR=SNR,stage=name_of_model_random)
            plot_architecture(path=main_path,
                            Name_of_model= os.path.join("random_selections", name_of_model_random, ""), stage="stage_3")
            plt.close()
        finally:
            print("done")
            # sys.stdout.close()
            # sys.stdout = original_stdout
            end_time_1 = time.time()
            timing_lines = [
        f"stage_1 took {_fmt(end_time_1 - start_time_1)} (hh:mm:ss)"]
        for line in timing_lines:
            print(line)

        with open(os.path.join(path, "timing.txt"), "w") as f:
            f.write("\n".join(timing_lines) + "\n")

        # torch.save(os.path.join(path, "outputs", "worst_BER.pt"), worst_BER)
        # torch.save(os.path.join(path, "outputs", "best_BER.pt"), best_BER)
        # torch.save(os.path.join(path, "outputs", "SNR.pt"), SNR)
        # plt.close()
