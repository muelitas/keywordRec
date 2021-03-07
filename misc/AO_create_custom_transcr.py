'''/**************************************************************************
    File: AO_create_custom_transcr.py
    Author: Mario Esparza
    Date: 03/03/2021
    
    Iterate through AOLME's folder. Grab transcripts (of ensligh or spanish,
    whichever specified) from each folder. Iterate through each transcript.
    Read their lines, grab wav_path and text. Assess if wav file has specified
    sample rate and if it is one channel. If so, save wav path and text of
    such in the new transcript. Do this for each wav file.
    
***************************************************************************''' 
import os
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

def get_transcript_path(folder_path):
    '''Iterate through files in {folder_path} until transcript is found'''
    for file in sorted(os.listdir(folder_path), reverse=True):
        if file.endswith('.csv'):
            file_path = folder_path + '/' + file
            return file_path
        
    print(f"ERROR: Couldn't find a transcript here {folder_path}")
    sys.exit()
    
spanish = True #Set to false if dealing with english
root = '/media/mario/audios'
src_dir = root + '/aolme_orig'
dst_dir = root + '/AO_SP'
transcr_path = dst_dir + '/transcript.txt'
SR = 16000

#If {dst_dir} exists, ask if okay to overwrite; otherwise, create it
check_folder(dst_dir)

#Get transcripts from aolme folder
print("Started scanning AOLME's folders...")    
transcripts = []
for folder in sorted(os.listdir(src_dir)):
    folder_path = src_dir + '/' + folder
    if folder.endswith('Spanish') and spanish: #if dealing wit spanish
        transcripts.append(get_transcript_path(folder_path))
    elif not folder.endswith('Spanish') and not spanish: #If dealing with engl
        transcripts.append(get_transcript_path(folder_path))

print(f" ...Done. I found {len(transcripts)} transcripts.")

#Iterate through each transcript in {transcripts}
transcr = open(transcr_path, 'w')
print("I am scanning each transcript now...")
for transcript in sorted(transcripts):
    #Get full path to current directory and get name of current directory
    dir_path = '/'.join(transcript.split('/')[:-1])
    dir_name = dir_path.split('/')[-1]
    print(f"\tScanning transcript in '{dir_name}' folder...", end='')
    
    #Open trancript and get its lines
    f = open(transcript, 'r')
    lines = f.readlines()
    f.close()
    
    #Iterate through each line in {transcript}
    for idx, line in enumerate(lines[1:]): #[1:] to ignore first row (headers)
        audio_ID, _, _, text = line.strip().split(',')
        
        #Create new wav path
        new_path = dir_name.split('L',1)[1][3:]
        new_path += '_' + audio_ID.zfill(3) + '.wav'
        new_path = dst_dir + '/' + new_path
        
        #Get wave and sample rate
        old_path = dir_path + '/' + audio_ID + '.wav'
        wave, sr = sf.read(old_path)
        
        #Check if audio's sample rate is the desired sample rate
        if sr != SR:
            print(f"ERROR: This audio {old_path} has a {sr} sample rate")
            sys.exit()
            
        #Ensure audio is one channel
        if len(wave.shape) != 1:
            print(f"ERROR: This file {old_path} doesn't have 1 channel")
            sys.exit()
            
        #If no errors, it is safe to copy wav file
        shutil.copy(old_path, new_path)
            
        #Save new line in new transcript
        transcr.write(new_path + '\t' + text + '\n')
    
    print(" ...Finished")
    
transcr.close()
print(f"Finished, all {len(transcripts)} transcripts have been processed\n")
