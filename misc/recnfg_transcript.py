'''/**************************************************************************
    File: recnfg_transcript.py
    Author: Mario Esparza
    Date: 02/26/2021
    
    Given a transcript, reconfigure its paths to match the ones in the
    respective computer.
    
***************************************************************************''' 
#TODO implement boolean to save me from warning below
#WARNING: this code will overwrite your current transcript. You may want to
#first print out some samples and comment the second loop.
transcr_path = '/home/mario/Desktop/ctc_data/spctrgrms/clean/TI_all_test/transcript.txt'
new_root = '/home/mario/Desktop/ctc_data/spctrgrms/clean/TI_all_test' 

#Get lines from transcript and create 'new lines'
new_lines = []
transcr = open(transcr_path, 'r')
lines = transcr.readlines()
for idx, line in enumerate(lines):
    old_path, text, duration = line.split('\t')
    new_path = new_root + '/' + old_path.split('/')[-1]
    new_lines.append(new_path + '\t' + text + '\t' + duration)
    
     #Use these lines to print some samples
    # print(f"Old Line: {line}", end='')
    # print(f"New Line: {new_lines[idx]}", end='')
    # if idx >= 1:
    #     break
    
transcr.close()

#Overwrite transcript and save 'new lines'
transcr = open(transcr_path, 'w')
for line in new_lines:
    transcr.write(line)
    
transcr.close()

print("Done, your transcript now has paths matching your computer")
