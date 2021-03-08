'''/**************************************************************************
    File: gTTS_and_TTS_merge.py
    Author: Mario Esparza
    Date: 03/07/2021
    
    Merge audios from gTTS and TTS in one folder. Create a transcript with two
    columns: full wav path and word said in wav audio. Randomize locations of
    audios. Also, ensure each file has a 16,000 sample rate.
    
***************************************************************************''' 
import os
import random
import shutil
import soundfile as sf
import sys

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

SR = 16000
root = '/media/mario/audios/spanglish'
new_root = '/media/mario/audios/spanglish_custom'
new_transcr_path = new_root + '/transcript.txt'
wav_folders = ['/gtts_wav', '/Lupe_new', '/Mia_new', '/Miguel_new',
               '/Penelope_new']

#Ensure {new_root} exists. If it does, ask if okay to delete contents
check_folder(new_root)

#Set up paths for transcripts in {wav_folders}
transcr_paths = [root + i + '/transcript.txt' for i in wav_folders]

#Make sure transcripts exist; grab wav_paths from each transcript
wav_paths = []
for transcript in transcr_paths:        
    if not os.path.exists(transcript):
        print(f"\nERROR: this transcript '{transcript}' doesn't exist")
        sys.exit()
        
    with open(transcript, 'r') as F:
        for line in F:
            wav_path, _ = line.split('\t')
            wav_paths.append(wav_path)  
            
#Randomize!
random.seed(5)
random.shuffle(wav_paths)

#Copy files to new (merged) directory; save info in transcript
print("Started merging files ...")
new_transcr = open(new_transcr_path, 'w')
for idx, src in enumerate(wav_paths):
    #Ensure sample rate is {SR}
    _, sr = sf.read(src)
    if sr != SR:
        print(f"ERROR: Sample rate of this audio {src} isn't {SR}Hz")
        sys.exit()
    
    #Copy audio (rename destination file)
    word = src.split('/')[-1].split('.')[0]
    new_name = str(idx).zfill(5) + '_' + word + '.wav'
    dst = new_root + '/' + new_name
    shutil.copy(src, dst)
    new_transcr.write(dst + '\t' + word + '\n')
    
    if idx % 400 == 0:
        print(f"\t{idx+1}/{len(wav_paths)} files have been copied")
    
new_transcr.close()
print(f" ...finished, all {len(wav_paths)} audios have been merged")