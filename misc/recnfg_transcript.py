'''/**************************************************************************
    File: recnfg_transcript.py
    Author: Mario Esparza
    Date: 02/26/2021
    
    Given a transcript, reconfigure its paths to match the ones in the
    respective computer.
    
***************************************************************************''' 
#WARNING: this code will overwrite your current transcript. You may want to
#first print out some samples and comment the second loop.
transcr_path = '/media/mario/audios/spctrgrms/clean/KA/transcript.txt'
new_root = '/media/mario/audios/spctrgrms/clean/test' 

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
    if idx >= 1:
        break
    
transcr.close()

#Overwrite transcript and save 'new lines'
# transcr = open(transcr_path, 'w')
# for line in new_lines:
#     transcr.write(line)
    
# transcr.close()

print("Done, your transcript now has paths matching your computer")
