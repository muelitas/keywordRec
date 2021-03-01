'''/**************************************************************************
    File: TI_create_phonemes_dict.py
    Author: Mario Esparza
    Date: 02/27/2021
    
    Use Speech Commands' dataset to create a vocabulary dictionary. It creates
    2 different files: .txt file and .pickle file. Both contain the same info,
    however, first one is "human-friendly" (you can open it right away and see
    a word translation to phonemes). Second one is used during preprocessing.
    Its loaded in the program as a dictionary and fastens the process of
    translating each transcript's sentence to phonemes.
    
***************************************************************************''' 
import os
import pickle
import sys

from phonemizer import phonemize
from phonemizer.separator import Separator

root = '/media/mario/audios'
audios_dir = root + '/speech_commands_v2'
dict_txt = root + '/dict/sc_dict.txt'
dict_pickle = root + '/dict/sc_dict.pickle'

#If {dict_txt} or {dict_pickle} already exist, ask if ok to overwrite
if os.path.exists(dict_txt):
    #If user's answer is "y", no need to re-create file here, I do so later on
    #If user's answer isn't "y", stop program
    print(f"{dict_txt} already exists, is it okay if I overwrite it? [y/n]")
    if input().lower() != 'y':
        sys.exit()

if os.path.exists(dict_pickle):
    print(f"{dict_pickle} already exists, is it okay if I overwrite it? [y/n]")
    if input().lower() != 'y':
        sys.exit()

#Get vocabulary, list of unique words in Speech Commands' folder
print("Processing Speech Commands's Transcript...")
vocabulary = []
for idx, file in enumerate(sorted(os.listdir(audios_dir))):
    if '_' not in file and 'E' not in file:
        vocabulary.append(file)

print(f" ...Finished, all {idx+1} files have been scanned\n")

#Sort vocabulary
vocabulary = sorted(vocabulary)
print(f"I found {len(vocabulary)} unique words in the folder\n")

#Parameters for phonemizer
#language 'es-la' latin-america, 'es' spain, 'en-us' USA, 'en' british
L = 'en-us'
B_E = 'espeak' #back end
c_sep = Separator(phone='_', syllable='', word=' ') #custom separator

#Start 'phonemization'
print("'Phonemization' of words has started...")
Dictionary = {}
Dict = open(dict_txt, 'w')
for idx, word in enumerate(vocabulary):
    phones = phonemize(word, L, B_E, c_sep)[:-2]
    phones = phones.replace('_', ' ')
    
    #Give warning if word has an 'empty' phoneme
    for ph in phones.split(' '):
        if ph == '':
            print(f"\tWARNING: This word '{word}' has an empty phoneme")
    
    Dict.write(word + '\t' + phones + '\n')
    Dictionary[word] = phones
    
Dict.close()
print(f" ...Finished. All {idx+1} words have been phomemized.\n")

#Save {Dictionary} in pickle file
pickle.dump(Dictionary, open(dict_pickle, "wb"))
print(f".txt dictionary has been saved here {dict_txt}")
print(f".pickle dictionary has been saved here {dict_pickle}")
