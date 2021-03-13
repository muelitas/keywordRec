'''/**************************************************************************
    File: display_spctrgrms.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 03/12/2021
    
    Display spectrograms from a given list of paths.
    
***************************************************************************'''
import matplotlib.pyplot as plt
import torch

def plot_spctrgrm(title, spctrgrm):
    '''Plot spctrgrm with specified {title}'''
    fig, ax = plt.subplots()  # a figure with a single Axes
    # ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
    plt.show()

spctrgrms_paths = [
    '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/KAx4/batalla_arapiles_4552_0.pt',
    '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/KAx4/batalla_arapiles_4552_1.pt',
    '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/KAx4/batalla_arapiles_4552_2.pt',
    '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/KAx4/batalla_arapiles_4552_3.pt',
    '/home/mario/Desktop/ctc_data/spctrgrms/clean/KA/batalla_arapiles_4552.pt'
]

for item in spctrgrms_paths:
    spec = torch.load(item)
    title = f"Spectrogram of {item.split('/')[-1]}"
    plot_spctrgrm(title, spec)