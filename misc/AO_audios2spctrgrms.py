'''/**************************************************************************
    File: AO_audios2spctrgrms.py
    Author: Mario Esparza
    Date: 03/03/2021
    
    Convert AOLME's audios to spectrograms. Normalize values before saving.
    Calculate duration of audio and add it to new transcript. Check if audios
    don't have a {SR} sample rate.
    
***************************************************************************''' 
import os
import shutil
import sys
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram as MelSpec

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

root = '/media/mario/audios'
old_transcr_path = root + '/AO_EN/transcript.txt'
new_transcr_path = root + '/spctrgrms/clean/AO_EN/transcript.txt'
dst_dir = root + '/spctrgrms/clean/AO_EN'
SR = 16000 #Sample Rate
mels = 128

#If {dst_dir} exists, ask if okay to overwrite; otherwise, create it
check_folder(dst_dir)

print("Processing AOLME's audios...")
sr_coeff = SR / 1000 #(/1000) to save audio's duration in miliseconds
#Open transcripts (as read and write respectively)
old_transcr = open(old_transcr_path, 'r')
new_transcr = open(new_transcr_path, 'w')
#Iterate through lines in old transcript
for idx, line in enumerate(old_transcr):
    old_path, text = line.strip().split('\t')
    audio_name = old_path.split('/')[-1]
    wave, sr = torchaudio.load(old_path)
    audio_dur = int(wave.size()[1] / sr_coeff)
    
    #Check if audio's sample rate is the desired sample rate
    if sr != SR:
        print(f"ERROR: This audio {old_path} has a {sr} sample rate")
        sys.exit()
    
    #Get spectrogram, normalize it and save it
    spec = MelSpec(sample_rate=SR, n_mels=mels)(wave)
    spec = normalize_0_to_1(spec)
    new_path = dst_dir + '/' + audio_name[:-4] + '.pt'
    torch.save(spec, new_path)
    
    #Save new line in new transcript (audio's duration included)
    new_transcr.write(new_path + '\t' + text + '\t' + str(audio_dur) + '\n')
    
    if idx%150 == 0:
        print(f"\t{idx+1} audios processed...")
    
new_transcr.close()
old_transcr.close()
print(f"Finished, all {idx+1} audios have been processed\n")