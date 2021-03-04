'''/**************************************************************************
    File: dict_count_phonemes.py
    Author: Mario Esparza
    Date: 03/04/2021
    
    Scan lines in dictionary and get a count of phonemes. Also, determine
    list of unique phonemes.
    
***************************************************************************''' 
dict_txt = '/media/mario/audios/dict/ti_all_test_dict.txt'

phs_counter = {}

f = open(dict_txt, 'r')
lines = f.readlines()
for line in lines[:None]:
    word, phs = line.strip().split('\t')
    for ph in phs.split(' '):
        if ph not in list(phs_counter.keys()):
            phs_counter[ph] = 1
        else:
            phs_counter[ph] += 1
            
f.close()
