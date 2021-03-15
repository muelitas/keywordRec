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
    '/home/mario/Desktop/AO_SP_smallerHopLength/Apr12-D-Chaitu_q2_02-05_Spanish_001.pt',
    '/home/mario/Desktop/ctc_data/spctrgrms/clean/AO_SP/Apr12-D-Chaitu_q2_02-05_Spanish_001.pt'
]

for item in spctrgrms_paths:
    spec = torch.load(item)
    title = f"Spectrogram of {item.split('/')[-1]}"
    plot_spctrgrm(title, spec)