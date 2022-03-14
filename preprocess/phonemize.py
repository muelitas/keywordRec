'''/**************************************************************************
    File: phonemize.py
    Author: Mario Esparza
    Date: 03/09/2022
    Description: given an audio dataset, iterate through its transcripts and
        transcribe them into IPA phonemes. Save a 'vocabulary' in a .pickle
        file.
    Updates: N/A
    TODO list: 
    - Apply 'smart' engine that replies "did you mean?" when user inputs dataset
    name differently than expected (maybe with cer and levenshtein distance)
    - If pickle file already exists, ask if okay to overwrite it
    
***************************************************************************''' 
import os
import pickle
import sys

from phonemizer import phonemize
from phonemizer.separator import Separator

class Phonemize():
    def __init__(self):
        self.dataset_name = ""
        self.dataset_path = "" 
        self.path_to_phonemes = ""

        #Parameters for phonemizer
        #language 'es-la' latin-america, 'es' spain, 'en-us' USA, 'en' british
        self.language = 'en-us'
        self.backend = 'espeak' #back end
        self.separator = Separator(phone='_', syllable='', word=' ')

    def check_user_input(self, user_input):
        try:
            self.dataset_name, self.dataset_path, self.path_to_phonemes = \
                user_input.split(' ')[1:]
        except Exception as e:
            print(f"Error: {e}")
        
    def phonemize_dataset(self):
        if(self.dataset_name == "speech_commands"):
            self.phonemize_speech_commands()
        else:
            print("Did you mean?")

    def phonemize_speech_commands(self):
        #Get sorted list of unique words in Speech Commands' folder
        print("Phonemizing Speech Commands's...")
        vocabulary = []
        for idx, file in enumerate(sorted(os.listdir(self.dataset_path))):
            if '_' not in file and 'E' not in file:
                vocabulary.append(file)
        
        vocabulary = sorted(vocabulary)

        #Start 'phonemization'
        print(f"...I found {len(vocabulary)} unique words...")
        Dictionary = {}
        for idx, word in enumerate(vocabulary):
            phones = phonemize(word, self.language, self.backend, \
                self.separator)[:-2]
            
            #Give warning if word has an 'empty' phoneme
            for ph in phones.split(' '):
                if ph == '':
                    print(f"\tWARNING: This word '{word}' has an empty phoneme")
            
            Dictionary[word] = phones

        #Save {Dictionary} in pickle file
        pickle.dump(Dictionary, open(self.path_to_phonemes, "wb"))
        print(f"...Finished. All words have been phomemized and saved ", end="")
        print(f"here {self.path_to_phonemes}.")
