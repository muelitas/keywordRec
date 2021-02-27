'''/**************************************************************************
    File: TI_audios2spectrograms.py
    Author: Mario Esparza
    Date: 02/26/2021
    
    Use custom transcript with .wav paths to TIMIT's audios. Get spectrogram
    of each audio, normalize it and save it. Create new transcript with 3
    columns: path to spectrogram, text said in such, and duration of such.
    
***************************************************************************''' 
import os
import shutil
import sys
import torch
import torchaudio
from torchaudio import transforms as tforms

def normalize_0_to_1(matrix):
    """Normalize matrix to a 0-to-1 range; code from
    'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
    d1, d2, d3 = matrix.size()#original dimensions
    matrix = matrix.reshape(d1, -1)
    matrix -= matrix.min(1, keepdim=True)[0]
    matrix /= matrix.max(1, keepdim=True)[0]
    matrix = matrix.reshape(d1, d2, d3)
    return matrix

old_transcr_path = '/media/mario/audios/TI/transcript.txt'
save_dir = '/media/mario/audios/spctrgrms/clean/TI'
new_transcr_path = save_dir + '/transcript.txt'
SR = 16000 #Sample Rate
mels = 128

#if nonexistent, create folder where spectrograms and transcript will be saved;
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

print("Started processing TIMIT's audios...")    
sr_coeff = SR / 1000 #divide by 1000 to save audio's duration in miliseconds
#Open transcripts (as read and write respectively)
old_transcr = open(old_transcr_path, 'r')
new_transcr = open(new_transcr_path, 'w')
#Iterate through lines in old transcript
for idx, line in enumerate(old_transcr):
    old_path, text = line.strip().split('\t')
    audio_name = old_path.split('/')[-1]
    wave, sr = torchaudio.load(old_path)
    audio_dur = int(wave.size()[1] / sr_coeff)
    
    #No need to check audio's sample rate since 'TI_get_lines_with_kwords.py'
    #already did it for me.
    
    #Get spectrogram, normalize it and save it
    spec = tforms.MelSpectrogram(sample_rate=SR, n_mels=mels)(wave)
    spec = normalize_0_to_1(spec)
    new_path = save_dir + '/' + audio_name[:-4] + '.pt'
    torch.save(spec, new_path)
    
    #Save .pt path, text and duration of audio in new transcript
    new_transcr.write(new_path + '\t' + text + '\t' + str(audio_dur) + '\n')
            
new_transcr.close()
old_transcr.close()
print(f" ...Finished, all {idx+1} audios have been processed. Spectrograms "
      f"and new transcript can be found here: {save_dir}.")