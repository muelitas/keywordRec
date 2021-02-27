'''/**************************************************************************
    File: TI_get_lines_with_kwords.py
    Author: Mario Esparza
    Date: 02/26/2021
    
    Grab all .TXT paths of 'TEST' and 'TRAIN' TIMIT folders. Use these .TXT
    files to determine if a keyword is present. If so, keep text and convert
    respective .WAV file to a .wav file (with 16,000 sampling rate) using
    ffmpeg.
    
***************************************************************************''' 
from glob import glob
import os
import shutil
import subprocess as subp
import sys

'''NOTES: I found some typos in resulting transcript:
For this line "but in this one section we welcomed auditors", I added 'the'
before 'auditors'. For this line: "one of the most desirable features for park
are beautiful views or scenery", I added 'a' before 'park. From this line:
"a sailboat may have a bone in her teeth one minute and lie becalmed the next"
I removed the 'a' before 'sailboat'.'''

timit_dir = '/media/mario/audios/TIMIT'
folders = ['TEST', 'TRAIN']
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers']
save_dir = '/media/mario/audios/TI'
new_transcr_path = save_dir + '/transcript.txt'
SR = 16000 #desired sample rate

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

#TODO combine the two outer-loops below
#Get all .TXT paths present in 'TEST' and 'TRAIN'
txt_paths = []
for folder in folders:
    txt_paths += glob(timit_dir + '/' + folder + '/*/*/*.TXT')

#Iterate throgh .TXT files, convert from WAV to wav and create new transcript
print("Iterating through .TXT paths...")
new_transcr = open(new_transcr_path, 'w')
counter = 0
for idx, txt_path in enumerate(txt_paths):
    txt_file = open(txt_path, 'r')
    line = txt_file.readlines()[0].strip().lower()
    
    #Remove special characters
    line = line.replace('.', '')
    line = line.replace(',', '')
    line = line.replace('?', '')
    line = line.replace(':', '')
    line = line.replace(';', '')
    line = line.replace('-', ' ')
    line = line.replace('mr ', 'mister ')
    #Get list of words in line (ignore start and end 'sampling times')
    words = line.split(' ')[2:]
    txt_file.close()
    
    #Iterate through {words}
    for word in words:
        if word in k_words:
            #Get old wav path and specify new wav path
            WAV_path = txt_path[:-4] + '.WAV'
            wav_path = save_dir + '/ti_' + str(counter).zfill(3) + '.wav'
            counter += 1
            #Save wav path and text in new transcript
            line = ' '.join(words)
            #Remove instances in which line has 2 or 3 contiguous spaces
            line = line.replace('   ', ' ')
            line = line.replace('  ', ' ')
            new_transcr.write(wav_path + '\t' + line + '\n')
            #Use ffmpeg to copy and convert WAV to wav with {SR} sampling rate
            cmd = f"ffmpeg -hide_banner -loglevel error -i '{WAV_path}'" 
            cmd += f" -acodec pcm_s16le -ac 1 -ar {SR} {wav_path}"
            subp.run(cmd, shell=True)
            break
        
    if idx%500 == 0:
        print(f"{idx+1}/{len(txt_paths)} txt files have been processed")
    
# new_transcr.close()
print(" ...Finished iterating through .TXT paths. New audios and transcript "
      f"have been save here: {save_dir}")