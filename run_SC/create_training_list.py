'''/**************************************************************************
    File: create_training_list.py
    Author(s): Mario Esparza
    Date: 03/18/2021
    
    Given the validation and testing lists from Speech Commands, produce a
    third list with the audios not included in the previous two. This will
    be our training list.
    
***************************************************************************''' 

import os

root = '/media/mario/audios/sc_v2_redownload'
dev_txt = root + '/validation_list.txt'
test_txt = root + '/testing_list.txt'
train_txt = root + '/training_list.txt'

#Read lines from validation list
old_dev = open(dev_txt, 'r')
dev_lines = old_dev.readlines()
old_dev.close()

#Read lines from testing list
old_test = open(test_txt, 'r')
test_lines = old_test.readlines()
old_test.close()

#Get names of folders from {root}
src_folders = []
for file in sorted(os.listdir(root)):
    if '_' not in file and 'E' not in file:
        src_folders.append(file)
        
#Create trainint list
train = open(train_txt, 'w')
general_counter = 0
train_counter = 0
print("Creating Training .txt file...")
for folder in src_folders:
    print(f"\tIn folder {folder}", end='')
    folder_path = root + '/' + folder
    for audio in os.listdir(folder_path):
        general_counter += 1
        line = folder + '/' + audio + '\n'
        if line not in dev_lines and line not in test_lines:
            train_counter+=1
            train.write(line)
            
    print(" ...Finished")
    
train.close()

print(f"Number of audios in Dev Dataset: {len(dev_lines)}")
print(f"Number of audios in Test Dataset: {len(test_lines)}")
print(f"Number of audios in Train Dataset: {train_counter}")
print(f"Total number of audios: {general_counter}")
