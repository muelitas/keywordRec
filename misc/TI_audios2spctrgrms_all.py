'''/**************************************************************************
    File: TI_audios2spctrgrms_all.py
    Author: Mario Esparza
    Date: 03/01/2021
    
    Iterate through lines in TIMIT's custom transcript. Grab wav path and text,
    determine audios duration in miliseconds; calculate (and save) spectrogram;
    and generate a new transcript with three columns: path of spectrogram,
    text said in such, and audio's duration.
    
    Custom transcript comes from TI_WAVs2wavs_all.py.
    
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

old_transcr_path = '/media/mario/audios/TI_all_test/transcript.txt'
save_dir = '/media/mario/audios/spctrgrms/clean/TI_all_test'
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

#Iterate through lines in old transcript
print("Started processing audios...")
old_transcr = open(old_transcr_path, 'r')
new_transcr = open(new_transcr_path, 'w')
lines = old_transcr.readlines()
sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in milisecs
for idx, line in enumerate(lines):
    #Grab wav_path and text said in such
    wav_path, text = line.strip().split('\t')  
    
    #Specify spectrogram path
    spctrgrm_path = save_dir + '/ti_' + str(idx).zfill(4) + '.pt'
    
    #Get wave; calculate and normalize spectrogram
    wave, _ = torchaudio.load(wav_path)
    spctrgrm = MelSpec(sample_rate=SR, n_mels=mels)(wave)
    spctrgrm = normalize_0_to_1(spctrgrm)
    
    #Save spec path, text and audios duration in new transcript
    audio_dur = int(wave.size()[1] / sr_coeff) # audio's duration
    new_transcr.write(spctrgrm_path + '\t' + text + '\t' + str(audio_dur)+'\n')
    torch.save(spctrgrm, spctrgrm_path)
        
    if idx%500 == 0:
        print(f"{idx+1}/{len(lines)} audios processed")
    
new_transcr.close()
old_transcr.close()
print(" ...Finished iterating through .TXT paths. Spectrograms and "
      f"transcript have been saved here: {save_dir}")
