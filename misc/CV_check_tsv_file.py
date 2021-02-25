'''/**************************************************************************
    File: CV_check_tsv_file.py
    Author: Mario Esparza
    Date: 02/24/2021
    
    Make sure each line in tsv file has a '\t'.
    
***************************************************************************''' 

tsv_path = '/media/mario/audios/en/final_train.tsv'

#Make sure each line in tsv file has a '\t'
tsv = open(tsv_path, 'r')
for line in tsv:
    if '\t' not in line:
        print(f"This line {line} doesn't have a '\\t'")
    
tsv.close()