import os
import time
import cProfile
import pstats
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))

from code_Networks.Network_multy_channels import Network_multy_channel,load_model
from code_Networks.SmallFunctions import plot_architecture, crateNetworkConnaction, compere_all_stages, plotSNRvsBER, \
    save_var_with_name, plot_symbols,_fmt
from code_Networks.Train_funcatios_clude import *
from code_GUI.GUI_input import save_params_to_file, parse_params_txt
import shutil



def clone_folder(src, dst):
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Cloned '{src}' → '{dst}'")
    except Exception as e:
        print(f"Clone Error: {e}")
# Parameters0
torch.manual_seed(20602026)



orignal_Name_of_model = "/home/dsi/mishans1/projects/code_for_server_based/simulation/multy_anntena/multy_anntena"


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")



for idx in range(1,5):
    path = os.path.join(os.pardir, f"{orignal_Name_of_model}_{idx}","")
    params = parse_params_txt(os.path.join(path, "parameters.txt"))
    N_users = params["N_users"]
    N_channels = params["N_channels"]
    Name_of_model = params["Name_of_model"]
    start_from_zero= False
    random_network = params["random_network"]
    N_transmitter = params["N_transmitter"]
    N_relays = params["N_relays"]
    N_users = params["N_users"]
    N_channels = params["N_channels"]
    N_rx      = params["N_rx"]
    N_tx      = params["N_tx"]
    demod_type = "simple"
    d_spacing = params["d_spacing"]
    SNR_basic_trainning=params["SNR_basic_trainning"]
    SNR_max=params["SNR_max"]
    max_iteration=params["max_iteration"]
    SNR_step=params["SNR_step"]
    BER_th=params["BER_th"]
    epochs=params["epochs"]
    z0_type=params["z0_type"]
    z0_init=params["z0_init"]
    z0_end=params["z0_end"]
    B=params["B"]
    sub_stages=params["sub_stages"]

    total_users = sum(N_users)
    modCode_order = [2 ** N for N in N_users]
    sub_model_path = os.path.join(path, "QAM baseline", "QAM baseline")
    model = load_model(path)
    model.demod_type = "simple"
    for c in range(model.N_channels):
        model.sub_networks[c].demod_type = "simple"
    does_stage_1 = True
    does_stage_2 = True
    does_stage_3 = True


    if not os.path.exists(sub_model_path):
        start_from_zero = False
        does_stage_1 = True
        does_stage_2 = True
        does_stage_3 = True
        os.makedirs(os.path.join(sub_model_path))
        os.makedirs(os.path.join(sub_model_path, "models"))
        os.makedirs(os.path.join(sub_model_path, "models", "python"))
        os.makedirs(os.path.join(sub_model_path, "models", "matlab"))
        os.makedirs(os.path.join(sub_model_path, "outputs"))
        os.makedirs(os.path.join(sub_model_path, "data"))
        clone_folder(os.path.join(path, "data"), os.path.join(sub_model_path, "data"))
    torch.save(demod_type, os.path.join(sub_model_path, "data", "demod_type.pt"))
    save_var_with_name(demod_type, "demod_type", os.path.join(sub_model_path, "data"))

    start_time_1 = time.time()
    if model.demod_type == "simple" and model.N_tx > 1 :
        print("2^M QAM deos not setcurrent more then 1 tx antenna")

    max_snr_train = stage_1(model=model,
                            basic_training=does_stage_1,
                            SNR_basic_trainning=SNR_basic_trainning,
                            SNR_max=SNR_max,
                            SNR_step=SNR_step,
                            max_iteration=max_iteration,
                            BER_th=BER_th,
                            device=device,
                            path=sub_model_path)
    # if does_stage_1:
    #     plotSNRvsBER(model=model,
    #                 num_itr=5,
    #                 batch=10 ** 4,
    #                 SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
    #                 stage="Stage 1"
    #                 )
    #     plt.savefig(os.path.join(sub_model_path, "outputs", "stage_1.png"))
    #     plt.clf()
    # model.save(sub_model_path,"stage_1")
    end_time_1 = time.time()

    start_time_2 = time.time()
    if does_stage_2:
        stage_2(model=model,
                max_snr_train=max_snr_train + 2,
                epochs=epochs,
                device=device,
                path=path,
                z0_type=z0_type,
                z0_init=z0_init,
                z0_end=z0_end,
                B=B,
                sub_stages=sub_stages
                )
        # plotSNRvsBER(model=model,
        #             num_itr=5,
        #             batch=10 ** 4,
        #             SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
        #             stage="Stage 2"
        #             )
        # plt.savefig(os.path.join(sub_model_path, "outputs", "stage_2.png"))
        # plt.clf() 
    else:
        model.load(sub_model_path,"stage_2")
    model.save(sub_model_path, "stage_2")

    end_time_2 = time.time()


  
    plot_architecture(path=path,Name_of_model=os.path.join("QAM baseline","QAM baseline",""))
    start_time_3 = time.time()
    if does_stage_3:
        model.culc_p()
        stage_3(model=model,
                SNR_basic_trainning=SNR_basic_trainning,
                SNR_max=max_snr_train+5,
                BER_th=BER_th,
                device=device,
                SNR_step=SNR_step,
                max_iteration=max_iteration,
                path=sub_model_path,
                SNR_val=max_snr_train)
        # plotSNRvsBER(model=model,
        #             num_itr=5,
        #             batch=10 ** 4,
        #             SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
        #             stage="Stage 3"
        #             )
        # plt.savefig(os.path.join(sub_model_path, "outputs", "stage_3.png"))
        # plt.clf()
    else:
        model.load(sub_model_path,"stage_3")
    model.save(sub_model_path, "stage_3")

    end_time_3 = time.time()




    plot_symbols(model=model,path = os.path.join(sub_model_path, ""))
    # compere_all_stages(model=model, path=os.path.join(sub_model_path, ""), batch_size=int(10**4),
    #                 SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21), num_itr=100)


    timing_lines = [
            f"stage_1 took {_fmt(end_time_1 - start_time_1)} (hh:mm:ss)",
            f"stage_2 took {_fmt(end_time_2 - start_time_2)} (hh:mm:ss)",
            f"stage_3 took {_fmt(end_time_3 - start_time_3)} (hh:mm:ss)",
        ]

    for line in timing_lines:
        print(line)

    with open(os.path.join(path, "timing.txt"), "w") as f:
        f.write("\n".join(timing_lines) + "\n")
    plt.close()
