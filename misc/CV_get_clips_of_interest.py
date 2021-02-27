'''/**************************************************************************
    File: CV_get_clips_of_interest.py
    Author: Mario Esparza
    Date: 02/24/2021
    
    Program that scans custom .tsv file from common voice (CV) dataset. Given
    a list of keywords, it grabs n number of clips (lines) of each keyword. It
    creates a .txt file for each keyword and saves these lines there. If a
    sentence has 2 or more keywords, it assigns it to the first keyword found.
    
***************************************************************************''' 
import os
import shutil
import sys

root = '/media/mario/audios/en'
save_path = root + '/clips_of_interest'
tsv_path = root + '/final_train.tsv'
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers']
n = 200 #Desired number of instances for each word
mp3_paths = []

#If folder where lines of interest will be saved doesn't exist, create it
if not os.path.isdir(save_path):
        os.mkdir(save_path)    
#If it exists, ask if okay to delete its contents
if len(os.listdir(save_path)) != 0:
    print(f"{save_path} isn't empty, is it okay if I overwrite it? [y/n]")
    if input().lower() != 'y':
        sys.exit()
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

#Read and scan through lines in tsv file
tsv = open(tsv_path, 'r')
tsv_lines = tsv.readlines()[1:] #[1:] to ignore header line
for k_word in k_words:
    k_word_txt_path = save_path + f'/{k_word}.txt'
    k_word_txt = open(k_word_txt_path, 'w')
    counter = 0
    for line in tsv_lines:
        try:
            mp3_path, text = line.strip().split('\t')
        except ValueError:
            print(f"This line gave me a value error: {line}")
            continue
            
        for word in text.split(' '):
            if word == k_word:
                #Make sure clips don't repeat
                if mp3_path not in mp3_paths:
                    mp3_paths.append(mp3_path)
                    k_word_txt.write(line)
                    counter += 1
                    break
                
        if counter >= n:
            break
    
    k_word_txt.close()
    print(f"{counter} lines were saved for '{k_word}' here {k_word_txt_path}")
    
tsv.close()

#Personal Note: After this, I check the texts of each clip and see if it
#contains any word from a different language. If it does, I remove the line
#from the transcript. I am saving these 'clean' transcripts in:
#'audios/CV/clips_of_interest_clean'.
