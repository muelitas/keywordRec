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
from os.path import join as pj #pj stands for path.join
from pathlib import Path
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
            title = "Spectrogram from Pyroom's Numpy Array"
            plot_spctrgrm(title, spctrgrm)
            
            #Get spectrogram using original audio and plot it
            spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(orig_samples)
            spctrgrm = normalize_0_to_1(spctrgrm)
            title = "Spectrogram from Original Audio"
            plot_spctrgrm(title, spctrgrm)
                        
    print(f"...finished, all {idx+1} spectrograms have been created. Samples of"
          " original audios and samples of pyroom audios have been plotted.\n")
    
    F.close()

def TS_npy_to_spctrgrms(tts_gtts, SR, mels):
    '''Use resulting numpy array from Pyroomacoustics to generate respective
    spectrograms. Use original audios to determine original lengths, since
    rows from numpy array were 0-padded to the longest audio in the dataset'''
    print(f"Started processing {tts_gtts['npy_file'].split('/')[-1]}...")
    sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
    
    #Grab audio waves from {npy_file} and transcript lines from {transcript}
    np_array = np.load(tts_gtts['npy_file'])
    f = open(tts_gtts['old_transcr'], 'r')
    lines = f.readlines()
    f.close()
    
    #Ensure {lines} and {np_array} have the same number of rows (lines)
    if len(lines) != np_array.shape[0]:
        print("ERROR: transcript and npy_file do not share the same # of rows\n")
        sys.exit()
    
    #Iterate through each line in old transcript
    F = open(tts_gtts['new_transcr'], 'w')
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
        spctrgrm_path = tts_gtts['dst_dir'] + '/' + filename[:-4] + '.pt'
        
        #Save spectrogram and save information in new transcript
        torch.save(spctrgrm, spctrgrm_path)
        F.write(spctrgrm_path + '\t' + text + '\t' + str(audio_dur) + '\n')
        
        if idx%1000 == 0:
            print(f"\t{idx+1} spectrograms have been processed and saved")
        
        #Print samples for visual aid (to ensure audios match rows in np_array)
        if idx == 0 or idx == 1 or idx == len(lines)-1:
            #Plot spectrogram that came from numpy array (from pyroom)
            title = "Spectrogram from Pyroom's Numpy Array"
            plot_spctrgrm(title, spctrgrm)
            
            #Get spectrogram using original audio and plot it
            spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(orig_samples)
            spctrgrm = normalize_0_to_1(spctrgrm)
            title = "Spectrogram from Original Audio"
            plot_spctrgrm(title, spctrgrm)
                        
    print(f"...finished, all {idx+1} spectrograms have been created. Samples of"
          " original audios and samples of pyroom audios have been plotted.\n")
    
    F.close()

def TI_npy_to_spctrgrms(timit, SR, mels):
    '''Use resulting numpy array from Pyroomacoustics to generate respective
    spectrograms. Use original audios to determine original lengths, since
    rows from numpy array were 0-padded to the longest audio in the dataset'''
    print(f"Started processing {timit['npy_file'].split('/')[-1]}...")
    sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
    
    #Grab audio waves from {npy_file} and transcript lines from {transcript}
    np_array = np.load(timit['npy_file'])
    f = open(timit['old_transcr'], 'r')
    lines = f.readlines()
    f.close()
    
    #Ensure {lines} and {np_array} have the same number of rows (lines)
    if len(lines) != np_array.shape[0]:
        print("ERROR: transcript and npy_file do not share the same # of rows\n")
        sys.exit()
    
    #Iterate through each line in old transcript
    F = open(timit['new_transcr'], 'w')
    for idx, line in enumerate(lines):
        #Grab samples from npy file
        audio_samples_np = np_array[idx]
        
        #Samples from npy file are zero padded; these 3 lines will unpad them.
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
        spctrgrm_path = timit['dst_dir'] + '/' + filename[:-4] + '.pt'
        
        #Save spectrogram and save information in new transcript
        torch.save(spctrgrm, spctrgrm_path)
        F.write(spctrgrm_path + '\t' + text + '\t' + str(audio_dur) + '\n')
        
        if idx%150 == 0:
            print(f"\t{idx+1} spectrograms have been processed and saved")
        
        #Print samples for visual aid (to ensure audios match rows in np_array)
        if idx == 0 or idx == 1 or idx == len(lines)-1:
            #Plot spectrogram that came from numpy array (from pyroom)
            title = "Spectrogram from Pyroom's Numpy Array"
            plot_spctrgrm(title, spctrgrm)
            
            #Get spectrogram using original audio and plot it
            spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(orig_samples)
            spctrgrm = normalize_0_to_1(spctrgrm)
            title = "Spectrogram from Original Audio"
            plot_spctrgrm(title, spctrgrm)
                        
    print(f"...finished, all {idx+1} spectrograms have been created. Samples of"
          " original audios and samples of pyroom audios have been plotted.\n")

def SC_npy_to_spctrgrms(speechCmds, SR, mels):
    '''Use resulting numpy array from Pyroomacoustics to generate respective
    spectrograms. Use original audios to determine original lengths, since
    rows from numpy array were 0-padded to the longest audio in the dataset'''
    print(f"Started processing {speechCmds['npy_dir'].split('/')[-1]}...")
    sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
    
    #Get names of folders from {src_dir}
    src_folders = []
    for file in sorted(os.listdir(speechCmds['src_dir'])):
        if '_' not in file and 'E' not in file:
            src_folders.append(file)
           
    #Make sure {npy_dir} and {src_dir} have same number of folders and share
    #the same name. If so, create such folder in {dst_dir}
    npy_files = sorted(os.listdir(speechCmds['npy_dir']))
    for src_folder, npy_file in zip(src_folders, npy_files):
        if src_folder not in npy_file:
            print("We have a mismatch in folder names, src_folder is: "
                  f"{src_folder} and npy_folder is {npy_file}")
            sys.exit()
        
        #If they match, create destination folder
        dst_folder = pj(speechCmds['dst_dir'], src_folder)
        os.mkdir(dst_folder)
            
    #
    for src_folder, npy_file in zip(src_folders, npy_files):
        print(f"\tWorking on {src_folder}...")
        #Specify paths
        old_dir_path = pj(speechCmds['src_dir'], src_folder)
        new_dir_path = pj(speechCmds['dst_dir'], src_folder)
        npy_dir_path = pj(speechCmds['npy_dir'], npy_file)
        
        #Grab numpy array from {npy_dir} and files in {old_dir}
        np_array = np.load(npy_dir_path)
        files = sorted(os.listdir(old_dir_path))
        
        #Ensure number of files and number of rows are the same
        if len(files) != np_array.shape[0]:
            print("ERROR: number of files and number of rows doesn't match "
                  f"for '{src_folder}'")
            sys.exit()
        
        #Itereate through files and rows
        new_transcr = open(pj(new_dir_path, 'transcript.txt'), 'w')
        for idx, file in enumerate(files):
            #Grab samples from npy array
            audio_samples_np = np_array[idx]
            
            #Samples from npy file are zero padded; this lines will unpad them
            audio_orig_path = pj(old_dir_path, file)
            text = src_folder
            orig_samples, _ = torchaudio.load(audio_orig_path)
            audio_dur = int(orig_samples.size()[1] / sr_coeff) #audio duration
            audio_samples_np = audio_samples_np[:orig_samples.size()[1]]
            
            #Convert from numpy array of integers to pytorch tensor of floats
            audio_samples_pt = torch.from_numpy(audio_samples_np)
            audio_samples_pt = torch.unsqueeze(audio_samples_pt, 0)
            audio_samples_pt = audio_samples_pt.type(torch.FloatTensor)
            
            #Calculate spectrogram and normalize
            spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(audio_samples_pt)
            spctrgrm = normalize_0_to_1(spctrgrm)
      
            #Get spectrogram path (where it will be saved)
            spctrgrm_path = new_dir_path + '/' + src_folder + '_'
            spctrgrm_path += str(idx).zfill(4) + '.pt'
            
            #Save spectrogram and save information in new transcript
            torch.save(spctrgrm, spctrgrm_path)
            new_transcr.write(f"{spctrgrm_path}\t{text}\t{audio_dur}\n")
            
            if idx%150 == 0:
                print(f"\t\t{idx+1} spectrograms have been processed")
                
            #Print samples for visual aid (to ensure files match rows)
            if idx == 0 or idx == len(files)-1:
                #Plot spectrogram that came from numpy array (from pyroom)
                title = f"{src_folder}'s spectrogram from Pyroom's Numpy Array"
                plot_spctrgrm(title, spctrgrm)
                
                #Get spectrogram using original audio and plot it
                spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(orig_samples)
                spctrgrm = normalize_0_to_1(spctrgrm)
                title = f"{src_folder}'s spectrogram from Original Audio"
                plot_spctrgrm(title, spctrgrm)
                        
        new_transcr.close()             
        print(f"...finished, {idx+1} spectrograms have been created. Samples "
              "of original and samples of pyroom have been plotted.")

    print(f"Finished processing all {len(src_folders)} folders")

data_root = str(Path.home()) + '/Desktop/ctc_data'
root = '/media/mario/audios'
SR = 16000 #Sample Rate
mels = 128

#KA stands for Kaggle
KA = {'old_transcr': root + '/spanish/kaggle_custom/transcript.txt',
      'new_transcr': data_root + '/spctrgrms/pyroom/KAx4_3/transcript.txt',
      'dst_dir': data_root + '/spctrgrms/pyroom/KAx4_3',
      'npy_file': root + '/PyRoom/KA_times4/KA_DA_3.npy'}

#TS stands for TTS and gTTS
TS = {'old_transcr': root + '/spanish_custom/transcript.txt',
      'new_transcr': '/media/mario/audios/TS_x4_spctrgrms/TSx4_3/transcript.txt',
      'dst_dir': '/media/mario/audios/TS_x4_spctrgrms/TSx4_3',
      'npy_file': '/media/mario/audios/PyRoom/TS_times4/TS_DA_3.npy'}

#TI stands for TIMIT
TI_train = {'old_transcr': root + '/TI_all_train/transcript.txt',
      'new_transcr': data_root + '/spctrgrms/pyroom/TI_all_train/transcript.txt',
      'dst_dir': data_root + '/spctrgrms/pyroom/TI_all_train',
      'npy_file': root + '/PyRoom/DA_files/TI_all_train_DA.npy'}

TI_test = {'old_transcr': root + '/TI_all_test/transcript.txt',
      'new_transcr': data_root + '/spctrgrms/pyroom/TI_all_test/transcript.txt',
      'dst_dir': data_root + '/spctrgrms/pyroom/TI_all_test',
      'npy_file': root + '/PyRoom/DA_files/TI_all_test_DA.npy'}

SC = {'src_dir': root + '/speech_commands_v2',
      'dst_dir': data_root + '/spctrgrms/pyroom/SC',
      'npy_dir': root + '/PyRoom/DA_files/SC_DA'}

#Check if destination directories exist, otherwise, ask if okay to erase
# check_folder(KA['dst_dir'])
check_folder(TS['dst_dir'])
# check_folder(LS['dst_dir'])
# check_folder(SC['dst_dir'])
# check_folder(TI_train['dst_dir'])
# check_folder(TI_test['dst_dir'])

#Generate and save spectrograms
# KA_npy_to_spctrgrms(KA, SR, mels)
TS_npy_to_spctrgrms(TS, SR, mels)
# LS_npy_to_audios(LS, SR, some_zeros)
# SC_npy_to_spctrgrms(SC, SR, mels)
# TI_npy_to_spctrgrms(TI_train, SR, mels)
# TI_npy_to_spctrgrms(TI_test, SR, mels)
