'''/**************************************************************************
    File: create_new_transcripts.py
    Author: Mario Esparza
    Date: 03/18/2021
    
    Create test, train and validations lists. Those that will have the paths
    to spectrograms (instead of audios).
    
***************************************************************************'''
import os
from os.path import join as pj #pj stands for path.join
import sys

old_root = '/media/mario/audios/sc_v2_redownload'
new_root = '/media/mario/audios/sc_v2_redownload_spctrgrms'

lists_names = sorted(['validation_list', 'testing_list', 'training_list'])

old_paths = [pj(old_root, i) + '.txt' for i in lists_names]
new_paths = [pj(new_root, i) + '.txt' for i in lists_names]

for old_list, new_list in zip(old_paths, new_paths):
    new_txt = open(new_list, 'w')
    old_txt = open(old_list, 'r')
    for old_line in old_txt:
        new_line = old_line[:-4] + 'pt'
        
        if not os.path.exists(pj(new_root, new_line)):
            print(f"This path {pj(new_root, new_line)} doesn't exist")
            sys.exit()
        
        new_txt.write(new_line + '\n')
        
    old_txt.close()
    new_txt.close()
    