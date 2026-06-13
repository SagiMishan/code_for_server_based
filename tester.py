import os
import time
import cProfile
import pstats
import io

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from Network_multy_channels import Network_multy_channel, load_model
from SmallFunctions import plot_architecture, crateNetworkConnaction, compere_all_stages,plotSNRvsBER,save_var_with_name
from GUI_input import get_parameters_gui_grouped,save_params_to_file
# Parameters

Name_of_model = "model_3_comper_results_single_TX_RX_anntenas"

path = ".\\" + Name_of_model + "\\"
sub_model_path = path + "QAM model\\QAM model"

demod_type = "simple"
torch.save(demod_type, sub_model_path + "\\data\\demod_type.pt")
save_var_with_name(demod_type, "demod_type", sub_model_path + "\\data\\")