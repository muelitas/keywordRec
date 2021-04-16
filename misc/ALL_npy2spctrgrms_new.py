'''/**************************************************************************
    File: npy_to_spctrgrms.py
    Author: Mario Esparza
    Date: 04/09/2021
    
    Given a dataset, use resulting python list from Pyroomacoustics to generate
    respective spectrograms. Save spectrograms in indicated folder, as well as a transcript
    with three columns: path to spectrogram, text said in such, and audio's
    duration in miliseconds. Refer to original spectrograms to make sure each
    filename is being assigned to the correct spectrogram.
    
***************************************************************************'''
import matplotlib.pyplot as plt
import os
import pickle
import shutil
import sys
import torch
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

def TS_npy_to_spctrgrms(tts, SR, mels):
    '''Use resulting numpy array from Pyroomacoustics to generate respective
    spectrograms. Use original audios to compare and make sure spectrograms
    match.'''
    print(f"Started processing {tts['pickle_file'].split('/')[-1]}...")
    #TODO calculate audio durations, right now I am grabbing old durations
    #durations that don't have the extra time from pyroom
    #Grab audio waves from {pickle_file} and transcript lines from {transcript}
    python_list = pickle.load(open(tts['pickle_file'], "rb" ))
    f = open(tts['old_transcr'], 'r')
    lines = f.readlines()
    f.close()
    
    #Ensure {lines} and {np_array} have the same number of rows (lines)
    if len(lines) != len(python_list):
        print("ERROR: transcript and pickle_file do not share the same # of rows\n")
        sys.exit()
    
    #Iterate through each line in old transcript
    F = open(tts['new_transcr'], 'w')
    for idx, line in enumerate(lines):
        #Grab samples from npy file
        audio_samples_np = python_list[idx]
        
        #Grab info from transcript
        audio_orig_path, text, _ = line.split('\t')
        
        #Convert from numpy array of doubles to pytorch tensor of floats
        audio_samples_pt = torch.from_numpy(audio_samples_np)
        audio_samples_pt = torch.unsqueeze(audio_samples_pt, 0)
        audio_samples_pt = audio_samples_pt.type(torch.FloatTensor)
        
        #Get duration of pyroom's audio. Multiply by 1000 to save as milisecs
        audio_dur = str(int((audio_samples_pt.size()[1] / SR) * 1000)) + '\n'
        
        #Calculate spectrogram and normalize
        spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(audio_samples_pt)
        spctrgrm = normalize_0_to_1(spctrgrm)
        
        #Get spectrogram path (where it will be saved)
        filename = audio_orig_path.split('/')[-1]
        spctrgrm_path = tts['dst_dir'] + '/' + filename[:-3] + '.pt'
        
        #Save spectrogram and save information in new transcript
        torch.save(spctrgrm, spctrgrm_path)
        F.write(spctrgrm_path + '\t' + text + '\t' + audio_dur)
        
        if idx%1000 == 0:
            print(f"\t{idx+1} spectrograms have been processed and saved")
        
        #Print samples for visual aid (to ensure audios match rows in np_array)
        if idx == 0 or idx == 1 or idx == len(lines)-1:
            #Plot spectrogram that came from numpy array (from pyroom)
            title = "Pyroom's Spectrogram"
            plot_spctrgrm(title, spctrgrm)
            
            #Get original spectrogram and plot it
            title = "Original Spectrogram"
            spctrgrm = torch.load(audio_orig_path)
            plot_spctrgrm(title, spctrgrm)
                       
    print(f"...finished, all {idx+1} spectrograms have been created. Samples of"
          " original audios and samples of pyroom audios have been plotted.\n")
    
    F.close()
    
SR = 16000 #Sample Rate
mels = 128

#TS stands for TTS and gTTS
TS = {#Reading from transcript that has paths to spectrograms
      'old_transcr': '/home/mario/Desktop/ctc_data/spctrgrms/clean/TS_SP_Phrases/transcript.txt',
      'new_transcr': '/media/mario/audios/TS_SP_phrases_x4_specs/TS_3/transcript.txt',
      'dst_dir': '/media/mario/audios/TS_SP_phrases_x4_specs/TS_3',
      'pickle_file': '/media/mario/audios/PyRoom/TS_SP_Phrases/TS_SP_Phrases_DA_3.npy.p'}

#Check if destination directories exist, otherwise, ask if okay to erase
# check_folder(KA['dst_dir'])
check_folder(TS['dst_dir'])

#Generate and save spectrograms
# KA_npy_to_spctrgrms(KA, SR, mels)
TS_npy_to_spctrgrms(TS, SR, mels)
