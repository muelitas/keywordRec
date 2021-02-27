'''/**************************************************************************
    File: sc_recnfg_transcript.py
    Author: Mario Esparza
    Date: 02/27/2021
    
    Given the sc folder, grab transcripts. Iterate through their lines and
    re-configure the paths saved in them, to match the ones in the
    respective computer.
    
***************************************************************************''' 
import os
from os.path import join as pj #pj stands for path.join
#WARNING: this code will overwrite your current transcripts. You may first want
#to print some samples and comment the second loop in the function.

def recnfg_transcript(new_root, transcr_path):
    #Get lines from transcript and create 'new lines'
    new_lines = []
    transcr = open(transcr_path, 'r')
    lines = transcr.readlines()
    for idx, line in enumerate(lines):
        old_path, text, duration = line.split('\t')
        new_path = new_root + '/' + old_path.split('/')[-1]
        new_lines.append(new_path + '\t' + text + '\t' + duration)
        
        #Use these lines to print some samples
        print(f"Old Line: {line}", end='')
        print(f"New Line: {new_lines[idx]}", end='')
        if idx >= 0:
            break
        
    transcr.close()
    
    #Overwrite transcript and save 'new lines'
    # transcr = open(transcr_path, 'w')
    # for line in new_lines:
    #     transcr.write(line)
        
    # transcr.close()
    print(f"-{transcr_path.split('/')[-2]}'s transcript has been reconfigured\n")

src_dir = '/media/mario/audios/spctrgrms/clean/SC'
new_root = '/media/mario/audios/spctrgrms/test/SC' 

#Iterate through folders in {src_dir}; grab a transcript and edit it
for folder in sorted(os.listdir(src_dir)):
    old_path = pj(src_dir, folder)
    new_path = pj(new_root, folder)
    recnfg_transcript(new_path, pj(old_path, 'transcript.txt'))

print("Done, all transcripts of Speech Commands have been reconfigured")
