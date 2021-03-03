'''/**************************************************************************
    File: audios2npy.py
    Author: Mario Esparza
    Date: 03/03/2021
    
    Read lines from custom transcript. Grab wav_path from each line, read the
    file (get the wave) from such and save it in a row of a numpy array. In
    other words, we prepare the audios listed in the transcript to be inputted
    in our Pyroomacoustics code. We zero pad all audios with repsect to the
    longest audio in the transcript.
    
***************************************************************************''' 
import numpy as np
import os
import soundfile as sf
import sys

transcript = '/media/mario/audios/TI_all_train/transcript.txt'
save_file = '/home/mario/Desktop/TI_all_train.npy'
SR = 16000 #Desired sample rate for all audios

#Check if {save_file} already exists. Ask if okay to continue.
if os.path.exists(save_file):
    print(f"This file {save_file} already exists. If we continue, it will be "
          "overwritten...do you want to continue? [y/n] ")
    if input() != 'y':
        sys.exit()

f = open(transcript, 'r')
lines = f.readlines()
# lines = lines[:5000]
f.close()

#Determine number of audios in transcript as well as the shape of longest audio
print("Determining longest audio...")
audios_lengths = []
for idx, line in enumerate(lines):
    wav_name = line.split('\t')[0]
    data, samplerate = sf.read(wav_name)
    
    #Ensure sample rate of audio is the desired sample rate
    if samplerate != SR:
        print(f"ERROR: This file {wav_name} doesn't have a {SR} sample rate")
        sys.exit()
        
    #Ensure audio is one channel
    if len(data.shape) != 1:
        print(f"ERROR: This file {wav_name} doesn't have 1 channel")
        sys.exit()

    audios_lengths.append(data.shape[0])
    
    if idx%500 == 0:
        print(f"\t[{idx+1} / {len(lines)}] audios have been scanned")

#Initialize matrix that will hold wave samples of all audios (zero padded)
max_len = max(audios_lengths)
X = np.zeros((len(lines), max_len)) 

print("\nSaving audios in matrix...")
for idx, line in enumerate(lines):
    wav_name = line.split('\t')[0]
    data, _ = sf.read(wav_name)
    X[idx][0:data.shape[0]] = data
    if idx%1000 == 0:
        print(f"\t[{idx+1} / {len(lines)}] audios have been added to matrix")

#Save matrix in .npy file
np.save(save_file, X)
print(f"\nNumpy array has been created and saved here {save_file}. It is "
      f"composed of {len(lines)} rows and {max_len} columns.")