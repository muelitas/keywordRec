'''/**************************************************************************
    File: npy_to_spctrgrms.py
    Author: Mario Esparza
    Date: 02/28/2021
    
    Given a dataset, use resulting numpy array from Pyroomacoustics to generate
    respective spectrograms. Refer to the original audios to determine original
    lengths, since rows in numpy array were 0-padded to the longest audio in
    the dataset. Save spectrograms in indicated folder, as well as a transcript
    with three columns: path to spectrogram, text said in such, and audio's
    duration in miliseconds.
    
***************************************************************************''' 
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram as MelSpec

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
            
def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range; code from
    'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
    d1, d2, d3 = matrix.size() #original dimensions
    matrix = matrix.reshape(d1, -1)
    matrix -= matrix.min(1, keepdim=True)[0]
    matrix /= matrix.max(1, keepdim=True)[0]
    matrix = matrix.reshape(d1, d2, d3)
    return matrix

def KA_npy_to_spctrgrms(kaggle, SR, mels):
    '''Use resulting numpy array from Pyroomacoustics to generate respective
    spectrograms. Use original audios to determine original lengths, since
    rows from numpy array were 0-padded to the longest audio in the dataset'''
    print(f"Started processing {kaggle['npy_file'].split('/')[-1]}...")
    sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
    
    #Grab audio waves from {npy_file} and transcript lines from {transcript}
    np_array = np.load(kaggle['npy_file'])
    f = open(kaggle['old_transcr'], 'r')
    lines = f.readlines()
    f.close()
    
    #Ensure {lines} and {np_array} have the same number of rows (lines)
    if len(lines) != np_array.shape[0]:
        print("ERROR: transcript and npy_file do not share the same # of rows\n")
        sys.exit()
    
    #Iterate through each line in old transcript
    F = open(kaggle['new_transcr'], 'w')
    for idx, line in enumerate(lines):
        #Grab samples from npy file
        audio_samples_np = np_array[idx]
        
        #Samples from npy file are zero padded; these 3 lines will 'unpad'.
        audio_orig_path, text = line.strip().split('\t')
        orig_samples, _ = torchaudio.load(audio_orig_path)
        audio_dur = int(orig_samples.size()[1] / sr_coeff) # audio's duration
        audio_samples_np = audio_samples_np[:orig_samples.size()[1]]
        
        #Convert from numpy array of integers to pytorch tensor of floats
        audio_samples_pt = torch.from_numpy(audio_samples_np)
        audio_samples_pt = torch.unsqueeze(audio_samples_pt, 0)
        audio_samples_pt = audio_samples_pt.type(torch.FloatTensor)
        
        #Calculate spectrogram and normalize
        spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(audio_samples_pt)
        spctrgrm = normalize_0_to_1(spctrgrm)
        
        #Get spectrogram path (where it will be saved)
        filename = audio_orig_path.split('/')[-1]
        spctrgrm_path = kaggle['dst_dir'] + '/' + filename[:-4] + '.pt'
        
        #Save spectrogram and save information in new transcript
        torch.save(spctrgrm, spctrgrm_path)
        F.write(spctrgrm_path + '\t' + text + '\t' + str(audio_dur) + '\n')
        
        if idx%150 == 0:
            print(f"\t{idx+1} spectrograms have been processed and saved")
        
        #Print samples for visual aid (to ensure audios match rows in np_array)
        if idx == 0 or idx == 1 or idx == len(lines)-1:
            #Plot spectrogram that came from numpy array (from pyroom)
            fig, ax = plt.subplots()  # a figure with a single Axes
            ax.set_title("Spectrogram from Pyroom's Numpy Array")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
            plt.show()
            
            #Get spectrogram using original audio and plot it
            spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(orig_samples)
            spctrgrm = normalize_0_to_1(spctrgrm)
            fig, ax = plt.subplots()  # a figure with a single Axes
            ax.set_title("Spectrogram from Original Audio")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
            plt.show()
            
    print(f"...finished, all {idx+1} spectrograms have been created. Samples of"
          " original audios and samples of pyroom audios have been plotted.\n")
    
    F.close()

def TI_npy_to_spctrgrms(timit, SR, zeros):
    ''' TODO '''
    print(f"Started processing {timit['npy_file'].split('/')[-1]}...")    
    #Grab audio waves from {npy_file} and transcript lines from {transcript}
    np_array = np.load(timit['npy_file'])
    f = open(timit['old_transcr'], 'r')
    lines = f.readlines()
    f.close()
    
    #Ensure {lines} and {np_array} have the same number of rows (lines)
    if len(lines) != np_array.shape[0]:
        print("ERROR: transcript and npy_file do not share the same # of rows\n")
        sys.exit()
    
    #Iterate through each to save new audios with respective name and phrase
    F = open(timit['new_transcr'], 'w')
    for idx, line in enumerate(lines):
        audio_samples = np_array[idx]
        # Trim zeros that might be at end of array (due to previoues padding)
        audio_samples = np.trim_zeros(audio_samples)
        #Add some zeros in case audio had some silence at the end
        audio_samples = np.append(audio_samples, zeros)
        
        old_path, phrase = line.split('\t')
        filename = old_path.split('/')[-1]
        new_path = timit['dst_dir'] + '/' + filename
        F.write(new_path + '\t' + phrase)
        # sf.write(new_path, audio_samples, SR)
        
        if idx%250 == 0:
            print(f'\t{idx+1} audios have been processed...')
        
    print(f"Finished...all {idx+1} audios have been created\n")
    F.close()

root = '/media/mario/audios'
SR = 16000 #Sample Rate
mels = 128

#KA stands for Kaggle
KA = {'old_transcr': root + '/spanish/kaggle_custom/transcript.txt',
      'new_transcr': root + '/spctrgrms/pyroom/KA/transcript.txt',
      'dst_dir': root + '/spctrgrms/pyroom/KA',
      'npy_file': root + '/PyRoom/DA_files/kaggle_DA.npy'}

#Check if destination directories exist, otherwise, ask if okay to erase
check_folder(KA['dst_dir'])
# check_folder(TS['dst_dir'])
# check_folder(LS['dst_dir'])
# check_folder(SC['dst_dir'])
# check_folder(TI['dst_dir'])

#Generate and save spectrograms
KA_npy_to_spctrgrms(KA, SR, mels)
# TS_npy_to_audios(TS, SR, some_zeros)
# LS_npy_to_audios(LS, SR, some_zeros)
# SC_npy_to_audios(SC, SR, some_zeros)
# TI_npy_to_audios(TI, SR, some_zeros)

