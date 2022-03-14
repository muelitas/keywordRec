'''/****************************************************************************
    File: preprocess/main.py
    Author(s): Mario Esparza
    Date: 03/12/2022
    Description: Given an audio dataset, process audios and return spectrograms
        as well as .csv with their respective paths and phrases spoken in the 
        audio.
    Updates: N/A
    TODO List:
    - Implement code to handle LibriSpeech, Commonvoice and TIMIT
    - For speech commands make sure self.num is not greater than lowest number
    of files in each one of its folders
    - Divide by 35 should only be done for speech commands
    - Apply 'smart' engine that replies "did you mean?" when user inputs dataset
    name differently than expected (maybe with cer and levenshtein distance)
    - Implement dynamic displaying of a few spectrograms as they are produced
    - Create one class per audio dataset
        
*****************************************************************************'''

import os
import pickle
import sys
import torch
import torchaudio

from os.path import join as pj
from random import shuffle as r_shuffle
from torchaudio.transforms import MelSpectrogram as MelSpec

class Preprocess_Data():
    def __init__(self):
        self.dataset_name = ""
        self.dataset_path = "" 
        self.ground_truth = "" #Path to ground truth folder
        self.csv_path = ""
        self.path_to_phonemes = ""
        self.num = None #number of audio files to use

        #Constants
        self.sr = 16000 #Sample Rate
        self.mels = 128
        self.thresh = 300 #Pad audio if duration is below {thresh} (miliseconds)
        self.zero_num = 4096 #Number of zeros padded to the right of the audio
        self.zero_pad = torch.nn.ZeroPad2d((0, self.zero_num, 0, 0)) #left, right, up, down

    def check_user_input(self, user_input):
        try:
            self.dataset_name, self.dataset_path, self.ground_truth, \
                self.path_to_phonemes = user_input.split(' ')[1:5]

            self.csv_path = self.ground_truth+"/gt.csv"

            if("-n" in user_input):
                self.num = int(user_input.split("-n ")[1].split(' ')[0]) // 35

        except Exception as e:
            print(f"Error: {e}")

    def preprocess_dataset(self):
        if(self.dataset_name == "speech_commands"):
            self.generate_specs_and_csvs_speech_commands()
        else:
            print("Did you mean?")

    def normalize_0_to_1(self, matrix):
        """Normalize matrix to a 0-to-1 range; code from
        'how-to-efficiently-normalize-a-batch-of-tensor-to-0-1' """
        d1, d2, d3 = matrix.size() #original dimensions
        matrix = matrix.reshape(d1, -1)
        matrix -= matrix.min(1, keepdim=True)[0]
        matrix /= matrix.max(1, keepdim=True)[0]
        matrix = matrix.reshape(d1, d2, d3)
        return matrix

    def check_audio_sample_rate(self, sr, audio_path):
        if sr != self.sr:
            print(f"ERROR: This audio {audio_path} has a {sr} sample rate")
            sys.exit()

    def check_audio_duration(self, num_of_samples):
        #Cehck if duration is below {thresh} miliseconds
        audio_dur = int(num_of_samples / (self.sr / 1000))
        if audio_dur < self.thresh:
            return True
        
        return False

    def get_files_list_speech_commands(self):
        folders = os.listdir(self.dataset_path)
        all_files_names = []
        for folder in folders:
            if '_' not in folder and 'E' not in folder:
                files_names = os.listdir(pj(self.dataset_path, folder))
                r_shuffle(files_names)
                
                for file_name in files_names[:self.num]:
                    all_files_names.append(pj(folder, file_name))

        return all_files_names
    
    def generate_specs_and_csvs_speech_commands(self):
        phonemes = pickle.load(open(self.path_to_phonemes, "rb"))

        files_paths = self.get_files_list_speech_commands()
        r_shuffle(files_paths)

        print("Generating Spectrograms and Csv Of Speech Commands...")
        gt = open(self.csv_path, "w")
        for idx, file_path in enumerate(files_paths):
            full_path = pj(self.dataset_path, file_path)
            wave, sr = torchaudio.load(full_path)

            #Perform a few checks
            self.check_audio_sample_rate(sr, full_path)
            if(self.check_audio_duration(wave.size()[1])):
                wave = self.zero_pad(wave) #pad {zero_num} miliseconds
                print(f"...changed audio duration for '{file_path}'")
    
            #Get spectrogram, normalize it and save it
            spec = MelSpec(sample_rate=self.sr, n_mels=self.mels)(wave)
            spec = self.normalize_0_to_1(spec)
            save_path = file_path.replace('wav', 'pt').replace('/', '_')
            save_path = pj(self.ground_truth, save_path)
            torch.save(spec, save_path)

            #Save spectrogram path and translation to phonemes in csv file
            gt.write(f"{save_path},{phonemes[file_path.split('/')[0]]}\n")

            if idx%150 == 0:
                print(f"...{idx+1}/{len(files_paths)} audios processed")
        
        gt.close()
        print(f"...finished. Csv has been saved here '{self.csv_path}'.")
