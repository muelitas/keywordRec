'''/**************************************************************************
    File: AO_audios2spctrgrms.py
    Author: Mario Esparza
    Date: 03/03/2021
    
    Convert AOLME's audios to spectrograms. Normalize values before saving.
    Calculate duration of audio and add it to new transcript. Check if audios
    don't have a {SR} sample rate. If audios duration is under a given
    threshold, pad it with given number of zeros.
    
***************************************************************************''' 
import matplotlib.pyplot as plt
from pathlib import Path
import os
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
            
data_root = str(Path.home()) + '/Desktop/ctc_data'
old_transcr_path = '/media/mario/audios/AO_SP/transcript.txt'
new_transcr_path = data_root + '/spctrgrms/clean/AO_SP/transcript.txt'
dst_dir = data_root + '/spctrgrms/clean/AO_SP'
SR = 16000 #Sample Rate
mels = 128
thresh = 300 #If audio's duration is below this number (miliseconds), pad it
zero_num = 4096 #Number of zeros padded to the right of the audio

#If {dst_dir} exists, ask if okay to overwrite; otherwise, create it
check_folder(dst_dir)

print("Processing AOLME's audios...")
sr_coeff = SR / 1000 #(/1000) to save audio's duration in miliseconds
#Open old transcript and get lines
old_transcr = open(old_transcr_path, 'r')
lines = old_transcr.readlines()
old_transcr.close()

#Zero pad function
zero_pad = torch.nn.ZeroPad2d((0, zero_num, 0, 0)) #left, right, up, down

#Iterate through lines in old transcript; save new ones in new_transcr
counter, print_sample = 0, False
new_transcr = open(new_transcr_path, 'w')
for idx, line in enumerate(lines):
    old_path, text = line.strip().split('\t')
    audio_name = old_path.split('/')[-1]
    wave, sr = torchaudio.load(old_path)
    
    #Check if audio's sample rate is the desired sample rate
    if sr != SR:
        print(f"ERROR: This audio {old_path} has a {sr} sample rate")
        sys.exit()
        
    #If duration is below {thresh} miliseconds, pad {zero_num} miliseconds
    audio_dur = int(wave.size()[1] / sr_coeff)
    if audio_dur < thresh:
        print(f"Audio #{idx+1}, duration changed from {audio_dur} ", end='')
        wave = zero_pad(wave)
        audio_dur = int(wave.size()[1] / sr_coeff)
        print(f"to {audio_dur}")
        counter += 1
        print_sample = True
    
    #Get spectrogram, normalize it and save it
    spec = MelSpec(sample_rate=SR, n_mels=mels)(wave)
    spec = normalize_0_to_1(spec)
    new_path = dst_dir + '/' + audio_name[:-4] + '.pt'
    torch.save(spec, new_path)
        
    #Save new line in new transcript (audio's duration included)
    new_transcr.write(new_path + '\t' + text + '\t' + str(audio_dur) + '\n')
    
    if idx%150 == 0:
        print(f"\t{idx+1}/{len(lines)} audios processed...")
        
    #To print some spectrograms' samples
    if counter > 0 and counter <=2 and print_sample:
        plot_spctrgrm(f"Audio #{idx+1} Spectrogram", spec)
        print_sample = False
    
new_transcr.close()

print(f"Finished, all {idx+1} audios have been processed. {counter} of them "
      "had to be zero padded. I plotted the first two zero-padded "
      "spectrograms for visual aid.")