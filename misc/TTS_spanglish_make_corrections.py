'''/**************************************************************************
    File: TTS_spanglish_make_corrections.py
    Author: Mario Esparza
    Date: 03/07/2021
    
    Synthesized audios from TTS may sound "robotic" or may be mispronounced.
    Given a list of words-to-remove and the directory where audios are, it
    copies the ones that are not in the list. Programs ensure each audio is
    formatted with a 16,000Hz sample rate. A transcript with two columns is
    created: full wav path and word spoken in such.
    
***************************************************************************''' 
from glob import glob
import os
from os.path import join as pj #pj stands for path.join
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

person = 'Penelope' #Choose from: Mia, Miguel, Penelope and Lupe
data_root = '/media/mario/audios/spanglish'
src_dir = pj(data_root, person)
dst_dir = pj(data_root, person + '_new')
transcr_path = pj(dst_dir, 'transcript.txt')
words2remove_path = pj(data_root, person + '_words2remove.txt')
SR = 16000

#If {dst_dir} exists, ask if okay to overwrite; otherwise, create it
check_folder(dst_dir)

#%% Grab audio paths; check for misspellings
# Create dictionary of word -> audio_path relation
wav_files = glob(src_dir + '/**.wav')
audios_paths = {}
for file in sorted(wav_files):
    word = file.split('/')[-1].split('_')[-1].split('.')[0]
    if word in list(audios_paths.keys()):
        print(f"This word '{word}' is repeated. Please review it.")
        sys.exit()
        
    audios_paths[word] = pj(src_dir, file)

# Grab words to remove from .txt file; make sure there aren't any mispellings
txt = open(words2remove_path, 'r')
lines = txt.readlines()
txt.close()

words2remove = []
for line in lines:
    line = line.strip()
    word = ''
    if '/' in line:
        word = line.split('/')[-1].split('_')[-1].split('.')[0]
    else:
        word = line
    
    if word not in list(audios_paths.keys()):
        print(f"This word '{word}' is misspelled. Please review it.")
        sys.exit()
        
    words2remove.append(word)

#%% Ensure sample rate is 16,000; copy audios
#Iterate through each audio, copy only the ones that aren't in {words2remove}
counter = 0
new_transcr = open(transcr_path, 'w')
for word, src_path in audios_paths.items():
    if word not in words2remove:
        #Make sure src audio has a {SR} sample rate
        _, sr = sf.read(src_path)
        if sr != SR:
            print(f"ERROR: this audio {src_path} doesn't have a {SR} sr.")
            sys.exit()
        
        #Copy audio and save its info in new transcript
        dst_path = pj(dst_dir, word + '.wav')
        shutil.copy(src_path, dst_path)
        new_transcr.write(dst_path + '\t' + word + '\n')
        counter += 1
        
new_transcr.close()

print(f"Originally, you had {len(list(audios_paths.keys()))} audios. From "
      f" those, {len(words2remove)} were removed and {counter} were copied.")
