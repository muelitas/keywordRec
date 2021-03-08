'''/**************************************************************************
    File: TSkw_merge_spctrgrms.py
    Author: Mario Esparza
    Date: 03/08/2021
    
    TTS_gTTS keywords; merge spectrograms that resulted from Pyroomascoustics.
    
***************************************************************************''' 
from glob import glob
import os
import random
import shutil
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

src_dir = '/media/mario/audios/PyRoom/TS_kwords'
dst_dir = '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/TS_kwords'

#Check if destination directory exists, otherwise, ask if okay to erase it
check_folder(dst_dir)

#Grab .txt transcripts from all subfolders in {src_dir}; and randomize them!
transcrs = glob(src_dir + '/*/*.txt')

#Iterate through lines in each transcript and keep them in {old_lines}
old_lines = []
for transcr in transcrs:
    with open(transcr, 'r') as f:
        for line in f:
            old_lines.append(line)

#Randomize their order
random.seed(7)
random.shuffle(old_lines)

#Iterate through {old_lines}; copy .pt file and add new info to new transcript
new_transcr = open(dst_dir + '/transcript.txt', 'w')
for idx, old_line in enumerate(old_lines):
    old_path, word, duration = old_line.split('\t')
    
    #Determine new_path; copy .pt from src to dst
    new_path = dst_dir + '/' + str(idx).zfill(3) + '_' + word + '.pt'
    shutil.copy(old_path, new_path)
    
    #Add new information to new transcript
    new_transcr.write(f"{new_path}\t{word}\t{duration}")
    
new_transcr.close()