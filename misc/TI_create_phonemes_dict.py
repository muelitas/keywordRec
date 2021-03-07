'''/**************************************************************************
    File: TI_create_phonemes_dict.py
    Author: Mario Esparza
    Date: 02/26/2021
    
    Use TIMIT's custom transcript to create a vocabulary dictionary. It creates
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
transcr_path = root + '/spctrgrms/clean/TI_all_train/transcript.txt'
dict_txt = root + '/dict/ti_all_train_dict.txt'
dict_pickle = root + '/dict/ti_all_train_dict.pickle'
#Parameters for phonemizer
#language 'es-la' latin-america, 'es' spain, 'en-us' USA, 'en' british
L = 'en-us'
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

#Get vocabulary, list of unique words in TIMIT's transcript
print("Processing TIMIT's Transcript...")
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
    
    #Manual edits
    if word == 'scouring' or word == 'lunchroom' or word == 'reupholstering' or word == 'brochure':
        if word == 'scouring':
            #Remove space between 'aɪʊ' and 'ɹ'
            phones = 's k aɪʊɹ ɪ ŋ'
        elif word == 'lunchroom':
            #Remove double space between 'tʃ' and 'ɹ'
            phones = 'l ʌ n tʃ ɹ uː m'
        elif word == 'reupholstering':
            #Change 'r' for 'ɹ'
            phones = 'ɹ j uː f ə l s t ɚ ɹ ɪ ŋ'
        else: #'brochure'
            #Remove 'r'
            phones = 'b ɹ oʊ ʃ ʊɹ'
    
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
import pickle
#Run this first to see the extra space
path_to_dict = '/home/mario/Desktop/ctc_data/dict/ti_all_test_dict.pickle'
Dictionary = pickle.load(open(path_to_dict, "rb" ))
print(Dictionary["lunchroom"])
#Then, run it together with this (with respective fix)
Dictionary["lunchroom"] = 'l ʌ n tʃ ɹ uː m'
print(Dictionary["lunchroom"])
pickle.dump(Dictionary, open(path_to_dict, "wb"))'''