'''/****************************************************************************
    File: main.py
    Author(s): Mario Esparza
    Date: 03/14/2022
    Description: Ask user for action, perform accordingly. Cases covered so far
        are Phonemization, Preprocessing and Training.
    Updates: N/A
    TODO List:
    - Add "testing" and "transfer learning"
    - Ask user if empty folder beforehand 
    - Log which GPU(s) were used
        
*****************************************************************************'''
import random
import torch

from preprocess.main import Preprocess_Data
from preprocess.phonemize import Phonemize
import train

word_to_stop_program = 'quit'
print(f"If you would like to stop, please type '{word_to_stop_program}'")

valid_commands = ["phonemize", "preprocess", "train"]
user_wants_to_exit = False

random.seed(7) #set random seed
torch.manual_seed(7)

while(not user_wants_to_exit):
    print("\nWhat are we doing today?")
    user_input = input()
    print("\n")
    command = user_input.split(' ')[0]
    
    if(command == word_to_stop_program):
        user_wants_to_exit = True
        print("Your wish is my command. Bye.")
        break
    
    if(command in valid_commands):
        if(command == "phonemize"):
            phonemes = Phonemize()
            phonemes.check_user_input(user_input)
            phonemes.phonemize_dataset()
        elif(command == "preprocess"):
            preprocessing = Preprocess_Data()
            preprocessing.check_user_input(user_input)
            preprocessing.preprocess_dataset()
        elif(command == "train"):
            trainer = train.Trainer()
            trainer.check_user_input(user_input)
            trainer.set_mappings_dictionaries(trainer.phonemes_path)
            trainer.initialize_params()
            trainer.initialize_datasets() #set 'train' and 'dev' datasets
            trainer.train()
    else:
        print(f"I am not sure what to do with this command: '{command}'.")
        print("Please read the docs to see valid commands")
        continue