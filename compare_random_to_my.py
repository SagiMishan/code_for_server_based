from os import listdir

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D


def get_high_and_low(path, ax, SNR, colors):
    BER_lowest = torch.ones((SNR.shape[0], 1))
    BER_high = torch.zeros((SNR.shape[0], 1))

    for random_model_name in listdir(path):
        print(random_model_name)
        worst_BER = torch.load(path + random_model_name + "\\outputs\\worst_BER.pt", weights_only=True)
        worst_BER = torch.unsqueeze(worst_BER.mean(dim=0), 1)
        BER_lowest = torch.minimum(worst_BER, BER_lowest)
        BER_high = torch.maximum(worst_BER, BER_high)
        # best_BER = torch.load(random_folder_path +random_model_name + "\\outputs\\best_BER.pt", weights_only=True)
        ax.semilogy(SNR, worst_BER, c=colors[0])

    # ax.fill_between(SNR, torch.squeeze(BER_lowest, dim=1), torch.squeeze(BER_high, dim=1), color=colors[1], alpha = 0.7)


Name_of_model = "simple_linear_network"

main_path = ".\\" + Name_of_model + "\\"
fig, ax = plt.subplots(figsize=(10, 6))

plt.style.use('classic')

# MATLAB-like color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])

# Use Helvetica-like font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

plt.title('Mean worst BER between the two channels ', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('BER (log scale)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

SNR = torch.load(main_path + "outputs\\SNR.pt", weights_only=True)

# get_high_and_low(path=main_path + "close_selections_distance_0\\", ax=ax, SNR=SNR, colors=['k', 'black'])
get_high_and_low(path=main_path + "close_selections_distance_3\\", ax=ax, SNR=SNR, colors=['r', 'red'])
# get_high_and_low(path=main_path + "close_selections_distance_2\\", ax=ax, SNR=SNR, colors=['y', 'yellow'])
get_high_and_low(path=main_path + "close_selections_distance_1\\", ax=ax, SNR=SNR, colors=['g', 'green'])

worst_BER = torch.load(main_path + "\\outputs\\worst_BER_stage_3.pt", weights_only=True)
ax.semilogy(SNR, worst_BER.mean(dim=0), c="b")

plt.xlim((SNR.min(), SNR.max()))

legend_elements = [

    Line2D([0], [0], color="r", lw=2, label=f"networks with Hamming distance = 3"),
    # Line2D([0], [0], color="y", lw=2, label=f"networks with Hamming distance = 2"),
    Line2D([0], [0], color="g", lw=2, label=f"networks with Hamming distance = 1"),
    # Line2D([0], [0], color="k", lw=2, label=f"networks with Hamming distance = 0"),
    Line2D([0], [0], color="b", lw=2, label=f" our solution ")
]

plt.legend(handles=legend_elements, loc="best")
plt.savefig(main_path + "outputs\\Mean worst BER between the two channels.png")
plt.show()
