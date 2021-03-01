'''/**************************************************************************
    File: CV_extract_clips.py
    Author: Mario Esparza
    Date: 02/25/2021
    
    Program that scans .txt files. It grabs the mp3 paths from this files and
    uses them to access audios in Common Voice's dataset. Audios are copied
    and converted to .wav (with 16,000 sampling rate) using ffmpeg. A new
    transcript is created containing two columns: full wav path and text said
    in such wav file.
    
***************************************************************************''' 
import os
import shutil
import subprocess as subp
import sys

audios_dir = '/media/mario/audios/CV/clips'
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers']
txts_dir = '/media/mario/audios/CV/clips_of_interest_clean'
save_dir = '/media/mario/audios/CV_custom'
new_transcr_path = save_dir + '/transcript.txt'
SR = 16000 #sample rate

#Create folder where new audios and transcript will be saved if non-existent
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

#Grab lines from all txt files
all_lines = []
for k_word in k_words:
    txt_file = open(txts_dir + '/' + k_word + '.txt', 'r')
    for line in txt_file:
        all_lines.append(line)
        
    txt_file.close()

#Sort {all_lines}
all_lines = sorted(all_lines)

#Iterate through {all_lines}; convert from mp3 to wav; create {new_transcr}
print("Starting conversion of audios...")
new_transcr = open(new_transcr_path, 'w')
for idx, line in enumerate(all_lines):
    mp3_path, text = line.split('\t')
    mp3_path = audios_dir + '/' + mp3_path
    wav_path = save_dir + '/cv_' + str(idx).zfill(4) + '.wav'
    #Use ffmpeg to copy and convert mp3 to wav
    cmd = "ffmpeg -hide_banner -loglevel error" #this removes ffmpeg verbose
    cmd += f" -i '{mp3_path}' -acodec pcm_s16le -ac 1 -ar {SR} {wav_path}"
    subp.run(cmd, shell=True)
    #Save wav_path and text in new transcript
    new_transcr.write(wav_path + '\t' + text)
    
    if idx%200 == 0:
        print(f"\t{idx+1}/{len(all_lines)} audios have been copied")
    
new_transcr.close()
print(" ...Conversion of audios finished. Audios and new transcript have been"
      f" saved here: {save_dir}.")