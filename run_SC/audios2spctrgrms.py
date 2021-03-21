'''/**************************************************************************
    File: audios2spctrgrms.py
    Author(s): Mario Esparza
    Date: 03/18/2021
    
    Get and save spectrograms of Speech Commands' audios.
    
***************************************************************************''' 
import matplotlib.pyplot as plt
import os
from os.path import join as pj #pj stands for path.join
import shutil
import sys
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram as MelSpec

def plot_spctrgrm(title, spctrgrm):
    '''Plot spctrgrm with specified {title}'''
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
    plt.show()

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range; code from
    'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
    d1, d2, d3 = matrix.size() #original dimensions
    matrix = matrix.reshape(d1, -1)
    matrix -= matrix.min(1, keepdim=True)[0]
    matrix /= matrix.max(1, keepdim=True)[0]
    matrix = matrix.reshape(d1, d2, d3)
    return matrix

def check_folder(this_dir):
    '''If {this_dir} exists, ask if okay to overwrite; otherwise, create it'''
    if not os.path.isdir(this_dir):
            os.mkdir(this_dir)    
            
    if len(os.listdir(this_dir)) != 0:
        print(f"{this_dir} isn't empty, is it okay if I overwrite it? [y/n]")
        if input().lower() != 'y':
            sys.exit()
        else:
            shutil.rmtree(this_dir)
            os.mkdir(this_dir)

src_root = '/media/mario/audios/sc_v2_redownload'
dst_root = '/media/mario/audios/sc_v2_redownload_spctrgrms'
SR = 16000
mels = 128

#Make sure destination directory exists (ask if okay to overwrite otherwise)
check_folder(dst_root)

#Get names of folders from {root}
src_folders = []
for file in sorted(os.listdir(src_root)):
    if '_' not in file and 'E' not in file:
        src_folders.append(file)
        
#Create subfolders in new root
src_folder_paths = []
dst_folder_paths = []
for idx, folder in enumerate(src_folders):
    os.mkdir(pj(dst_root, folder))
    dst_folder_paths.append(pj(dst_root, folder))
    src_folder_paths.append(pj(src_root, folder))

print(f"{idx+1} folders have been created in dst_root")

#Iterate through each audio file; create and save spectrogram for each
for src_path, dst_path in zip(src_folder_paths, dst_folder_paths):
    print(f"Processing audios of {src_path}...", end='')
    for file in os.listdir(src_path):
        file_src = pj(src_path, file)
        file_dst = pj(dst_path, file)
        
        #Load audio
        wave, sr = torchaudio.load(file_src)
    
        #Check if audio's sample rate is the desired sample rate
        if sr != SR:
            print(f"ERROR: This audio {file_src} has a {sr} sample rate")
            sys.exit()
            
        #Get spectrogram, normalize it and save it
        spec = MelSpec(sample_rate=SR, n_mels=mels)(wave)
        spec = normalize_0_to_1(spec)
        file_dst = file_dst[:-4] + '.pt'
        torch.save(spec, file_dst)
        
    print(" ... finished.")
    