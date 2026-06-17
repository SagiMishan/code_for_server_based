import os
import time
import cProfile
import pstats
import io

import numpy as np
import pandas as pd
import torch

from code_Networks.Network_multy_channels import Network_multy_channel
from code_Networks.SmallFunctions import plot_architecture, crateNetworkConnaction, compere_all_stages, plotSNRvsBER, \
    save_var_with_name, plot_symbols,_fmt
from code_Networks.Train_funcatios_clude import *
from code_GUI.GUI_input import save_params_to_file, parse_params_txt
# Parameters0


# source ./.venv/bin/activate
torch.manual_seed(20602026)

# init\general parameters
_GPU_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters_comp.txt")


params = parse_params_txt(_GPU_PARAMS_FILE)

orignal_Name_of_model = params["Name_of_model"]
start_from_zero= params["start_from_zero"]
random_network = params["random_network"]
does_stage_1 = params["does_stage_1"]
does_stage_2 = params["does_stage_2"]
does_stage_3 = params["does_stage_3"]
demod_type = params["demod_type"]
N_transmitter = params["N_transmitter"]
N_relays = params["N_relays"]
N_users = params["N_users"]
N_channels = params["N_channels"]
N_rx      = params["N_rx"]
N_tx      = params["N_tx"]

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

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
for idx in range(100):
    Name_of_model = f"{orignal_Name_of_model}_{idx}"
    if not os.path.exists(os.path.join(Name_of_model)):
        start_from_zero = True
        does_stage_1 = True
        does_stage_2 = True
        does_stage_3 = True
        os.makedirs(os.path.join(Name_of_model))
        os.makedirs(os.path.join(Name_of_model, "models"))
        os.makedirs(os.path.join(Name_of_model, "models", "python"))
        os.makedirs(os.path.join(Name_of_model, "models", "matlab"))
        os.makedirs(os.path.join(Name_of_model, "outputs"))
        os.makedirs(os.path.join(Name_of_model, "data"))

    path = os.path.join(Name_of_model, "")
    if start_from_zero:
        does_stage_1 = True
        does_stage_2 = True
        does_stage_3 = True
        if random_network:
            R = 1
            Phi = 90
            sector = 120

            connectaionMatrix, MatcgSR, MatcgRR, cgRU, cgTU, posR, posU, posT = crateNetworkConnaction(
                path=path,
                N_transmitter=N_transmitter,
                N_relays=N_relays,
                N_users=N_users,
                N_channels=N_channels,
                R=R,
                Phi=Phi,
                sector=sector,
                R_user=R * 1.1,
                N_rx=N_rx,N_tx=N_tx,
                d_spacing=d_spacing)
        else:
            aaa = torch.cat(
                [torch.zeros(int(N_relays / 2), int(N_relays / 2)), torch.ones(int(N_relays / 2), int(N_relays / 2))],
                dim=1)
            bbb = torch.cat(
                [torch.zeros(int(N_relays / 2), int(N_relays / 2)), torch.zeros(int(N_relays / 2), int(N_relays / 2))],
                dim=1)
            ccc = torch.cat([torch.ones(1, int(N_relays / 2)), torch.zeros(1, int(N_relays / 2))], dim=1)
            ddd = torch.cat([aaa, bbb], dim=0)
            eee = torch.cat([ddd, ccc], dim=0)

            connectaionMatrix = torch.cat([eee, torch.zeros(N_relays + 1, 1)], dim=1)
            connectaionMatrix = [connectaionMatrix, connectaionMatrix]

            # const 123

            cgRU = [torch.cat([torch.zeros(int(N_relays / 2),  N_users[c]), torch.ones(int(N_relays / 2), N_users[c])], dim=0) for c in range(N_channels)]
            # cgRU = torch.stack([cgRU, cgRU], dim=0)
            MatcgSR = torch.cat([torch.ones(int(N_relays / 2), 1), torch.zeros(int(N_relays / 2), 1)], dim=0)
            MatcgSR = torch.stack([MatcgSR.T, MatcgSR.T], dim=0)

            # F21_c1 = torch.tensor([[-0.5, 1, -0.5j, 1e-4, -0.5j * 1e-4, 0.5j * 1e-4],
            #                        [0.5, -0.5j, -1, -0.5e-4, 0.5j * 1e-4, -1e-4],
            #                        [1, -0.5, -0.5j, 0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4],
            #                        [1e-4, -0.5j * 1e-4, 0.5j * 1e-4, 1e-4, -0.5j * 1e-4, 0.5j * 1e-4],
            #                        [-0.5e-4, 0.5j * 1e-4, -1e-4, -0.5e-4, 0.5j * 1e-4, -1e-4],
            #                        [0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4, 0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4]]).T

            F21_c1 = torch.tensor([[-0.5, 1, -0.5j, 0, 0, 0],
                                [0.5, -0.5j, -1, 0, 0, 0],
                                [1, -0.5, -0.5j, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]).T
            ggg_c1 = torch.cat([torch.zeros(int(int(N_relays / 2)), int(N_relays / 2)), F21_c1], dim=1)
            MatcgRR_c1 = torch.cat([ggg_c1, torch.zeros(int(N_relays / 2), N_relays)], dim=0)

            # F21_c2 = torch.tensor([[1e-4, -0.5j * 1e-4, 0.5j * 1e-4, 1e-4, -0.5j * 1e-4, 0.5j * 1e-4],
            #
            #                        [-0.5e-4, 0.5j * 1e-4, -1e-4, -0.5e-4, 0.5j * 1e-4, -1e-4],
            #
            #                        [0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4, 0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4],
            #
            #                        [1e-4, -0.5j * 1e-4, 0.5j * 1e-4, -0.5, 1, -0.5j],
            #
            #                        [-0.5e-4, 0.5j * 1e-4, -1e-4, 0.5, -0.5j, -1],
            #
            #                        [0.5 * 1e-4, -0.5j * 1e-4, 1 * 1e-4, 1, -0.5, -0.5j]]).T

            F21_c2 = torch.tensor([[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, -0.5, 1, -0.5j],
                                [0, 0, 0, 0.5, -0.5j, -1],
                                [0, 0, 0, 1, -0.5, -0.5j]]).T
            ggg_c2 = torch.cat([torch.zeros(int(N_relays / 2), int(N_relays / 2)), F21_c2], dim=1)
            MatcgRR_c2 = torch.cat([ggg_c2, torch.zeros(int(N_relays / 2), N_relays)], dim=0)

            MatcgRR = torch.stack([MatcgRR_c1, MatcgRR_c2], dim=0)
            posR = torch.tensor([[1, 3], [1, 2], [1, 1], [1, -3], [1, -2], [1, -1],
                            [2, 3], [2, 2], [2, 1], [2, -3], [2, -2], [2, -1]])
            posT = torch.tensor([[-1, 1], [-1, -1]])
            posU = torch.stack([torch.tensor([[3, 2], [3, 1]]),torch.tensor([ [3, -1], [3, -2]])])
            N_tx = 1
            N_rx = 1
            N_relays = 12
            cgTU = [torch.tensor([0,0],dtype=torch.complex64).T,torch.tensor([0,0],dtype=torch.complex64).T]



        torch.save(MatcgRR, os.path.join(path, "data", "MatcgRR.pt"))
        save_var_with_name(MatcgRR,"MatcgRR",os.path.join(path, "data"))
        torch.save(MatcgSR, os.path.join(path, "data", "MatcgSR.pt"))
        save_var_with_name(MatcgSR,"MatcgSR",os.path.join(path, "data"))
        torch.save(cgRU, os.path.join(path, "data", "cgRU.pt"))
        save_var_with_name(cgRU,"cgRU",os.path.join(path, "data"))
        torch.save(cgTU, os.path.join(path, "data", "cgTU.pt"))
        save_var_with_name(cgTU, "cgTU", os.path.join(path, "data"))
        torch.save(connectaionMatrix, os.path.join(path, "data", "connectaionMatrix.pt"))
        save_var_with_name(connectaionMatrix,"connectaionMatrix",os.path.join(path, "data"))
        torch.save(N_users, os.path.join(path, "data", "N_users.pt"))
        save_var_with_name(torch.tensor(N_users), "N_users", os.path.join(path, "data"))
        torch.save(N_channels, os.path.join(path, "data", "N_channels.pt"))
        save_var_with_name(N_channels,"N_channels",os.path.join(path, "data"))
        torch.save(N_relays, os.path.join(path, "data", "N_relays.pt"))
        save_var_with_name(N_relays,"N_relays",os.path.join(path, "data"))
        torch.save(demod_type, os.path.join(path, "data", "demod_type.pt"))
        save_var_with_name(demod_type,"demod_type",os.path.join(path, "data"))
        torch.save(posT, os.path.join(path, "data", "posT.pt"))
        save_var_with_name(posT,"posT",os.path.join(path, "data"))
        torch.save(posR, os.path.join(path, "data", "posR.pt"))
        save_var_with_name(posR,"posR",os.path.join(path, "data"))
        torch.save(posU, os.path.join(path, "data", "posU.pt"))
        save_var_with_name(posU,"posU",os.path.join(path, "data"))
        torch.save(N_rx, os.path.join(path, "data", "N_rx.pt"))
        save_var_with_name(N_rx,"N_rx",os.path.join(path, "data"))
        torch.save(N_tx, os.path.join(path, "data", "N_tx.pt"))
        save_var_with_name(N_tx,"N_tx",os.path.join(path, "data"))
        torch.save(d_spacing, os.path.join(path, "data", "d_spacing.pt"))
        save_var_with_name(d_spacing,"d_spacing",os.path.join(path, "data"))
    else:
        MatcgRR = torch.load(os.path.join(path, "data", "MatcgRR.pt"), weights_only=True)
        MatcgSR = torch.load(os.path.join(path, "data", "MatcgSR.pt"), weights_only=True)
        cgRU = torch.load(os.path.join(path, "data", "cgRU.pt"), weights_only=True)
        cgTU = torch.load(os.path.join(path, "data", "cgTU.pt"), weights_only=True)
        connectaionMatrix = torch.load(os.path.join(path, "data", "connectaionMatrix.pt"), weights_only=True)
        N_users = torch.load(os.path.join(path, "data", "N_users.pt"), weights_only=True)
        N_channels = torch.load(os.path.join(path, "data", "N_channels.pt"), weights_only=True)
        N_relays = torch.load(os.path.join(path, "data", "N_relays.pt"), weights_only=True)
        demod_type = torch.load(os.path.join(path, "data", "demod_type.pt"), weights_only=True)
        posT = torch.load(os.path.join(path, "data", "posT.pt"), weights_only=False)
        posR = torch.load(os.path.join(path, "data", "posR.pt"), weights_only=False)
        posU = torch.load(os.path.join(path, "data", "posU.pt"), weights_only=False)
        N_tx = torch.load(os.path.join(path, "data", "N_tx.pt"), weights_only=True)
        N_rx = torch.load(os.path.join(path, "data", "N_rx.pt"), weights_only=True)

        d_spacing  = torch.load(os.path.join(path, "data", "d_spacing.pt"),  weights_only=True)
        modCode_order = [2 ** N for N in N_users]

    params["demod_type"] = demod_type
    params["N_transmitter"] = N_transmitter
    params["N_relays"] = N_relays
    params["N_users"] = N_users
    params["N_channels"] = N_channels
    params["N_rx"] = N_rx
    params["N_tx"] = N_tx
    params["d_spacing"]  = d_spacing
    save_params_to_file(params, path + "parameters.txt")


    start_time_1 = time.time()
    if demod_type == "simple" and N_tx > 1 :
        print("2^M QAM deos not setcurrent more then 1 tx antenna")
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
    max_snr_train = stage_1(model=model,
                            basic_training=does_stage_1,
                            SNR_basic_trainning=SNR_basic_trainning,
                            SNR_max=SNR_max,
                            SNR_step=SNR_step,
                            max_iteration=max_iteration,
                            BER_th=BER_th,
                            device=device,
                            path=path)
    # if does_stage_1:
    #     plotSNRvsBER(model=model,
    #                  num_itr=5,
    #                  batch=10 ** 5,
    #                  SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
    #                  stage="Stage 1"
    #                  )
    #     plt.savefig(os.path.join(path, "outputs", "stage_1.png"))
    #     plt.clf()
    model.save(path,"stage_1")
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
        #              num_itr=5,
        #              batch=10 ** 5,
        #              SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
        #              stage="Stage 2"
        #              )
        # plt.savefig(os.path.join(path, "outputs", "stage_2.png"))
        # plt.clf()
    else:
        model.load(path,"stage_2")
    model.save(path, "stage_2")

    end_time_2 = time.time()



    plot_architecture(path=path)
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
                path=path,
                SNR_val=max_snr_train)
        # plotSNRvsBER(model=model,
        #              num_itr=5,
        #              batch=10 ** 5,
        #              SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21),
        #              stage="Stage 3"
        #              )
        
        # plt.savefig(os.path.join(path, "outputs", "stage_3.png"))
        # plt.clf()
    else:
        model.load(path,"stage_3")
    model.save(path, "stage_3")

    end_time_3 = time.time()



    plot_symbols(model=model,path = path)
    # compere_all_stages(model=model, path=path, batch_size=int(10**4),
    #                    SNR=torch.linspace(max_snr_train - 10, max_snr_train + 10, 21), num_itr=100)


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