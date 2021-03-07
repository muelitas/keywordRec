'''/**************************************************************************
    File: utils.py
    Author(s): Mario Esparza
    Date: 03/06/2021
    
    TODO
    
***************************************************************************'''
import os
from pathlib import Path
import shutil
import sys

data_root = str(Path.home()) + '/Desktop/ctc_data' 
src_dir = data_root + '/spctrgrms/clean/TS'
dst_dir = data_root + '/spctrgrms/clean/TS_kwords'
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers', 'cero',
          'uno', 'dos', 'tres', 'cinco', 'número', 'números']

#Create folder where spectrograms and transcript will be saved if non-existent
if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)    
#If it exists, ask if okay to delete its contents
if len(os.listdir(dst_dir)) != 0:
    print(f"{dst_dir} isn't empty, is it okay if I overwrite it? [y/n]")
    if input().lower() != 'y':
        sys.exit()
    else:
        shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)

#Get lines from src_dir's transcript
transcr = open(src_dir + '/transcript.txt', 'r')
lines = transcr.readlines()
transcr.close()

#Iterate through eacn line and determine which do and don't have a k_word
lines_with_kword = []
lines_without_kword = []
for line in lines:
    pt_path, text, _ = line.split('\t')
    if text in k_words:
        lines_with_kword.append(line)
    else:
        lines_without_kword.append(line)
    
#Copy spectrograms in {lines_with_kword} to new dir; create respective transcr
transcr = open(dst_dir + '/transcript.txt', 'w')
for line in lines_with_kword:
    old_pt_path, text, duration = line.split('\t')
    new_pt_path = dst_dir + '/' + old_pt_path.split('/')[-1]
    shutil.copy(old_pt_path, new_pt_path)
    #Remove from old directory
    os.remove(old_pt_path)
    #Save line in new transcript
    transcr.write(f"{new_pt_path}\t{text}\t{duration}")

transcr.close()

#Overwrite old transcript (only with lines that have no keywords)
transcr = open(src_dir + '/transcript.txt', 'w')
for line in lines_without_kword:
    transcr.write(line)
    
transcr.close()