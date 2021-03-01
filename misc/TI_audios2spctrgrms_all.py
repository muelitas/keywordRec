'''/**************************************************************************
    File: TI_audios2spctrgrms_all.py
    Author: Mario Esparza
    Date: 03/01/2021
    
    Iterate through files in 
    
***************************************************************************''' 
from glob import glob
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

timit_dir = '/media/mario/audios/TIMIT/TRAIN'
save_dir = '/media/mario/audios/spctrgrms/clean/TI_all_train'
new_transcr_path = save_dir + '/transcript.txt'
SR = 16000 #desired sample rate
mels = 128

#Create folder where spectrograms and transcript will be saved if non-existent
if not os.path.isdir(save_dir):
        os.mkdir(save_dir)    
#If it exists, ask if okay to delete its contents
if len(os.listdir(save_dir)) != 0:
    print(f"{save_dir} isn't empty, is it okay if I overwrite it? [y/n]")
    if input().lower() != 'y':
        sys.exit()
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

#TODO combine the two outer-loops below
#Get all .TXT paths present in {timit_dir}
txt_paths = []
txt_paths += glob(timit_dir + '/*/*/*.TXT')

#Iterate throgh .TXT files ...
#TODO
print("Iterating through .TXT paths...")
transcr = open(new_transcr_path, 'w')
counter = 0
sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
for idx, txt_path in enumerate(txt_paths):
    #Grab text (phrase) from .txt file
    txt_file = open(txt_path, 'r')
    line = txt_file.readlines()[0].strip().lower().split(' ', 2)[-1]
    txt_file.close()
    
    #Remove special characters
    line = line.replace('.', '')
    line = line.replace(',', '')
    line = line.replace('?', '')
    line = line.replace(':', '')
    line = line.replace(';', '')
    line = line.replace('-', ' ')
    line = line.replace('mr ', 'mister ')
    line = line.replace('mrs ', 'missus ')    
    
    #Get wav path and specify spectrogram path
    WAV_path = txt_path[:-4] + '.WAV'
    spctrgrm_path = save_dir + '/ti_' + str(counter).zfill(4) + '.pt'
    counter += 1
    
    #Get wave; calculate and normalize spectrogram
    wave, _ = torchaudio.load(WAV_path)
    spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(wave)
    spctrgrm = normalize_0_to_1(spctrgrm)
    
    #Remove instances in which line has 2, 3 or 4 contiguous spaces
    line = line.replace('    ', ' ')
    line = line.replace('   ', ' ')
    line = line.replace('  ', ' ')
    
    #Save spec path, text and audios duration in new transcript
    audio_dur = int(wave.size()[1] / sr_coeff) # audio's duration
    transcr.write(spctrgrm_path + '\t' + line + '\t' + str(audio_dur) + '\n')
    torch.save(spctrgrm, spctrgrm_path)
        
    if idx%500 == 0:
        print(f"{idx+1}/{len(txt_paths)} txt files have been processed")
    
    if idx == 5:
        break
    
transcr.close()
print(" ...Finished iterating through .TXT paths. Spectrograms and "
      f"transcript have been saved here: {save_dir}")
