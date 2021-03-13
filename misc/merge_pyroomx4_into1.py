'''/**************************************************************************
    File: merge_pyroomx4_into1.py
    Author: Mario Esparza
    Date: 03/12/2021
    
    Given four different outputs of Pyroom, merge them all into one folder.
    
***************************************************************************''' 
import os
from os.path import join as pj #pj stands for path.join
import shutil

src_dir = '/media/mario/audios/TS_x4_spctrgrms'
dst_dir = '/home/mario/Desktop/ctc_data/spctrgrms/pyroom/TSx4'

src_folders = sorted(os.listdir(src_dir))
new_transcr_path = dst_dir + '/transcript.txt'

new_transcr = open(new_transcr_path, 'w')
for idx, folder in enumerate(src_folders):
    src_transcr_path = pj(src_dir, folder, 'transcript.txt')
    src_transcr = open(src_transcr_path, 'r')
    lines = src_transcr.readlines()
    src_transcr.close()

    print(f"Working on {folder}...")
    for line in lines:
        old_path, text, duration = line.split('\t')
        old_name = old_path.split('/')[-1].split('.')[0]
        new_path = pj(dst_dir, old_name + '_' + str(idx) + '.pt')
        shutil.copy(old_path, new_path)
        
        new_transcr.write(f"{new_path}\t{text}\t{duration}")
        
    print(f" ...Finished, all {len(lines)} audios have been copied")
    
new_transcr.close()