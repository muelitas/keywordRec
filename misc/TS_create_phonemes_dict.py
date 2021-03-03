'''/**************************************************************************
    File: TS_create_phonemes_dict.py
    Author: Mario Esparza
    Date: 03/02/2021
    
    Use TTS-gTTS's custom transcript to create a vocabulary dictionary. Create
    2 different files: .txt file and .pickle file. Both contain the same info,
    however, first one is "human-friendly" (you can open it right away and see
    a word translation to phonemes). Second one is used during preprocessing.
    Its loaded in the program as a dictionary and fastens the process of
    translating each transcript's sentence to phonemes.
    
***************************************************************************''' 
import pickle
import os
import sys

from phonemizer import phonemize
from phonemizer.separator import Separator

root = '/media/mario/audios'
dict_txt = root + '/dict/ts_dict.txt'
dict_pickle = root + '/dict/ts_dict.pickle'
transcr_path = root + '/spctrgrms/clean/TS/transcript.txt'
#Parameters for phonemizer
#language 'es-la' latin-america, 'es' spain, 'en-us' USA, 'en' british
L = 'es-la'
B_E = 'espeak' #back end
c_sep = Separator(phone='_', syllable='', word=' ') #custom separator

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

#Get vocabulary, list of unique words in TTS-gTTS's transcript
print("Processing TTS-gTTS's Transcript...")
vocabulary = []
transcr = open(transcr_path, 'r')
for idx, line in enumerate(transcr):
    _, text, _ = line.split('\t')
    words = text.split(' ')
    for word in words:
        if word not in vocabulary:
            #no need to apply .lower to {word}; already done in previous code
            vocabulary.append(word)
            
    if idx%100 == 0:
        print(f"\t{idx+1} lines scanned")
              
print(f" ...Finished, all {idx+1} lines have been scanned\n")
transcr.close()

#Sort vocabulary
vocabulary = sorted(vocabulary)
print(f"I found {len(vocabulary)} unique words in the transcript\n")

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

    if idx % 500 == 0:
        print(f"\t[{idx+1}/{len(vocabulary)}] words have been phonemized...")
    
Dict.close()
print(f" ...Finished. All {idx+1} words have been phomemized.\n")

#Save {Dictionary} in pickle file
pickle.dump(Dictionary, open(dict_pickle, "wb"))
print(f".txt dictionary has been saved here {dict_txt}")
print(f".pickle dictionary has been saved here {dict_pickle}")

'''Use this to fix any words that have an empty phoneme:
    #Run this first to see the extra space
    path_to_dict = '/home/mario/Desktop/ctc_data/dict/ti_all_test_dict.pickle'
    Dictionary = pickle.load(open(path_to_dict, "rb" ))
    print(Dictionary["lunchroom"])
    #Then, run it together with this (and respective fix)
    Dictionary["lunchroom"] = 'l ʌ n tʃ ɹ uː m'
    print(Dictionary["lunchroom"])
    pickle.dump(Dictionary, open(path_to_dict, "wb"))'''