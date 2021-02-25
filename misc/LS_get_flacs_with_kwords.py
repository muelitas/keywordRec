'''/**************************************************************************
    File: LS_get_flacs_with_kwords.py
    Author: Mario Esparza
    Date: 02/24/2021
    
    Given folders from LibriSpeech (LS), scan through subfolders and grab
    transcripts. Then, scan through lines in each transcript. Keep lines that
    include at least one instance of any keyword in the given list. Save these
    lines in a new 'custom' transcript.
    
***************************************************************************''' 
import os
import sys

#TODO use glob.glob to get transcripts
#NOTE Don't forget to remove words from other languages in resulting transcript
root = '/media/mario/audios/LibriSpeech'
folders_paths = [root + '/train-clean-100', root + '/train-clean-360']
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers']
new_transcr_path = root + '/custom_transcript.txt'

#Iterate through {transcrs_paths}, keep flacs that contain a keyword
new_transcr = open(new_transcr_path, 'w')
counter = 0 #Global counter of audios being kept
for folder_path in sorted(folders_paths):
    #...I am in train-clean-100, train-clean-360
    print(f'{folder_path}')
    for i, subF in enumerate(sorted(os.listdir(folder_path))):#subF=subfolder
        #...I am in 19, 26, 27, 32, etc.
        subF_path = folder_path + '/' + subF
        print(f'\tSubFolder [{i+1}/{len(os.listdir(folder_path))}]')
        for j, subsubF in enumerate(sorted(os.listdir(subF_path))):
            #...I am in 198, 227, 495, 496, etc.
            subsubF_path = subF_path + '/' + subsubF
            #Get transcript from sub-subfolder
            files = sorted(os.listdir(subsubF_path))
            transcript = files.pop()
            
            #Make sure it is the transcript (should be the last itme in files)
            transcr_path = subsubF_path + '/' + transcript
            if not transcr_path.endswith('trans.txt'):
                print(f"ERROR: This file {transcr_path} isn't a transcript. "
                      f"Check this folder: {subsubF_path}.")
                sys.exit()
            
            #Iterate through lines in transcript; keep line if k_word is found
            transcr = open(transcr_path, 'r')
            for line in transcr:
                flac_path, text = line.strip().split(' ', 1)
                text = text.lower()
                words = text.split(' ')
                for word in words:
                    if word in k_words:
                        #Add librispeech folder to flac path
                        flac_path = f"{subsubF_path.split('/')[-3]}_{flac_path}"
                        
                        #Save line in new transcript
                        new_transcr.write(f"{flac_path}\t{text}\n")
                        counter+=1
                        break
                    
            transcr.close()
            
    print()
    
new_transcr.close()       
     
print(f"{counter} lines have been found in these folders {folders_paths}. "
      "Each line includes at least one instance of one of the following "
      f"keywords: {k_words}. Lines have been saved here {new_transcr_path}.")