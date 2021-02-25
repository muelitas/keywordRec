'''/**************************************************************************
    File: LS_check_transcript.py
    Author: Mario Esparza
    Date: 02/24/2021
    
    Program that scans a custom version of LibriSpeech's (LS) transcript. It
    makes sure all files exist. It also makes sure that all the files listed
    in the transcript are the only ones in the given folder.
    
***************************************************************************''' 
import os

transcript = '/media/mario/audios/LibriSpeech_Custom/transcript.txt'
LS_folder = '/media/mario/audios/LibriSpeech_Custom' 

f = open(transcript, 'r')
for idx, line in enumerate(f):
    path, _ = line.split('\t')
    if not os.path.exists(path):
        print(f"ERROR: This file {path} doesn't exist :(")
    
f.close()

num_of_files = len(os.listdir(LS_folder)) -1 #-1 to remove transcript file

if(idx+1 != num_of_files):
    print("There is a mismatch in the number of files that you have in your "
          "folder and the number of files listed in your transcript. You "
          "should check it out.")