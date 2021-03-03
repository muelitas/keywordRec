'''/**************************************************************************
    File: TI_WAVs2wavs_all.py
    Author: Mario Esparza
    Date: 03/03/2021
    
    Iterate through all .TXT files in 'TRAIN' or 'TEST' of TIMIT's corpus.
    Grab text from each .txt file; determine respective .wav file; reconfigure
    to pcm and custom sample rate (using ffmpeg); lastly, generate a transcript
    with two columns: path to new wav file and text said in such.
    
***************************************************************************''' 
from glob import glob
import os
import shutil
import subprocess as subp
import sys

timit_dir = '/media/mario/audios/TIMIT/TEST'
save_dir = '/media/mario/audios/TI_all_test'
new_transcr_path = save_dir + '/transcript.txt'
SR = 16000 #desired sample rate

#Create folder where wavs and transcript will be saved if non-existent
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

#Get all .TXT paths present in {timit_dir}
txt_paths = []
txt_paths += glob(timit_dir + '/*/*/*.TXT')
txt_paths = sorted(txt_paths)

#Iterate throgh .TXT files
print("Iterating through .TXT paths...")
transcr = open(new_transcr_path, 'w')

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
    line = line.replace('"', ' ')
    line = line.replace('!', ' ')
    line = line.replace('mr ', 'mister ')
    line = line.replace('mrs ', 'missus ')    
    
    #Get WAV path and specify wav path
    WAV_path = txt_path[:-4] + '.WAV'
    wav_path = save_dir + '/ti_' + str(idx).zfill(4) + '.wav'
    
    #Use ffmpeg to re-configure wav file; ensure it is pcm and has {SR}Hz
    cmd = "ffmpeg -hide_banner -loglevel error" #this removes ffmpeg verbose
    cmd += f" -i '{WAV_path}' -acodec pcm_s16le -ac 1 -ar {SR} {wav_path}"
    subp.run(cmd, shell=True)
    
    #Remove contiguous white spaces as well as ending or starting ones
    line = line.replace('    ', ' ')
    line = line.replace('   ', ' ')
    line = line.replace('  ', ' ')
    line = line.strip()
    
    #Save new wav path and text in new transcript
    transcr.write(wav_path + '\t' + line + '\n')
        
    if idx%500 == 0:
        print(f"{idx+1}/{len(txt_paths)} txt files have been processed")
    
transcr.close()
print(" ...Finished iterating through .TXT paths. New wav files and "
      f"transcript have been saved here: {save_dir}")
