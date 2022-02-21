'''/**************************************************************************
    File: main_inference.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 03/03/2021
    
    Loads checkpoint (model) and performs inferences on specified dataset. 
    Calcuates PER and WER.
    
***************************************************************************'''
import os
from os.path import join as pj #pj stands for path.join
from pathlib import Path
import pickle
import random
import sys
import torch
import torch.nn.functional as F
import warnings

from models import SpeechRecognitionModel
from preprocessing import get_mappings
from utils import chars_to_int, GreedyDecoder, chars_to_ipa, cer, wer, \
    readablechars2IPA, IPA2customchars, log_message

#Comment this from time to time and check warnings are the same
warnings.filterwarnings("ignore")

##############################################################################
#VARIABLES THAT MIGHT NEED TO BE CHANGED ARE ENCLOSED IN THESE HASHTAGS
#Root location of logs, plots and checkpoints
runs_root = str(Path.home()) + '/Desktop/ctc_runs'
#Root location of spectrograms and dictionaries
data_root = str(Path.home()) + '/Desktop/ctc_data'

par_dir = 'ML2_KAx4_60E3Run3Rerun' #name of run
stg_dir = 'stg1'
chckpnt = 'checkpoint_onRun01onEpoch040.tar'
logfile = 'inferences_on_AO_sp_all.txt'

#Transcript that will be "inferenced" and the respective dictionary
transcr_path = data_root + '/spctrgrms/clean/AO_SP/transcript.txt'
dict_transcr = data_root + '/dict/ao_sp_dict.pickle'

#If you didn't train with all specs in transcript and are running inferences
#in the remaining, change these variables accordingly
start, end = 0, None

#Set to True, those dictionaries that were used in the checkpoint
dicts_chckpt = {'ka_dict': 1, #0=False, 1=True
                'ts_dict': 1,
                'ts_spang_dict': 0,
                'ao_sp_dict': False,
                'ao_en_dict': 0,
                'ti_all_train_dict': False,
                'ti_all_test_dict': False,
                'sc_dict': 0}

other_chars = [' ']
manual_chars = ['!','?','(',')','+','*','#','$','&','-','=',':']
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers', 'cero',
          'uno', 'dos', 'tres', 'cinco', 'número', 'números']
#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#Set up full paths to checkpoint and logfile
chckpnt_path = pj(runs_root, par_dir, stg_dir, chckpnt)
logfile_path = pj(runs_root, par_dir, stg_dir, logfile)

#Make sure directory where logfile is being saved exists
logfile_dir = '/'.join(logfile_path.split('/')[:-1])
if not os.path.exists(logfile_dir):
    print("The folder in which you want to save the inferences' results "
          "doesn't exist. Please double check it.")
    sys.exit()
    
#If {logfile} already exists, ask if okay to overwrite
if os.path.exists(logfile_path):
    print("The file in which you want to save the inferences already "
          "exist. You sure you want me to continue? [y/n]")
    if input().lower() != 'y':
        sys.exit()

#Grab only the dictionaries that were used for the checkpoint
dicts = [k for k, v in dicts_chckpt.items() if v]
#Add full path to each dictionaty in {dicts}
dicts = [f"{data_root}/dict/{item}.pickle" for item in dicts]
#Used {dicts} to get mappings between IPA, Custom Characters and Integers
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings([],
    other_chars, manual_chars, dicts)

#Determine if gpu is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
    
#Initialize seeds
torch.manual_seed(7)
random.seed(7)

#Ensure all phonemes in transcript exist in the phonemes which with the
#checkpoint was trained
words_dict = pickle.load(open(dict_transcr, "rb" ))
for word, phs_seq in words_dict.items():
    for ph in phs_seq.split(' '):
        if ph not in list(ipa2char.keys()):
            print(f"Checkpoint wasn't trained to recognize this phoneme {ph}")
            sys.exit()
    
#Load and prep checkpoint for evaluation
checkpnt = torch.load(chckpnt_path)
hparams = checkpnt['hparams']

#Ensure num_classes used in chckpnt match the num_classes specified here
if hparams['n_class'] != blank_label+1:
    print(f"\nThere is a mismatch in the number of classes. Checkpoint's is "
          f"{hparams['n_class']} while here, you have {blank_label+1}.")
    sys.exit()

model = SpeechRecognitionModel(hparams)
model.load_state_dict(checkpnt['model_state_dict'])
model = model.to(device)
model.eval()

#Read lines from transcript that will be used for inferences
transcr = open(transcr_path, 'r')
lines = transcr.readlines()[start:end]
transcr.close()

#Dictionary that keeps track of how many times each keyword is in transcript
k_words_num = {}
for k_word in k_words:
    k_words_num[k_word] = 0
    
#Iterate through lines, calculate PER and WER for each line
log = open(logfile_path, 'w')
log.write("Line#\tPER\tWER\tTarget IPAs\tPredicted IPAs\n")
with torch.no_grad():
    print("Running inferences now...")
    pers, wers = [], []
    for idx, line in enumerate(lines):
        #Get text and spectrogram
        spctrgrm_path, text, _ = line.split('\t')
        spec = torch.load(spctrgrm_path).unsqueeze(1)
        
        #If keywords are in text, count up
        for word in text.split(' '):
            if word in list(k_words_num.keys()):
                k_words_num[word] += 1
        
        #Text: from readable characters to IPA phonemes (separated by '_')
        text = readablechars2IPA(text, words_dict)
        #Text: from IPA to custom characters
        text = IPA2customchars(text, ipa2char)
        #Text: from custom characters to integers
        label = torch.Tensor(chars_to_int(text, char2int)).unsqueeze(0)
        label_length = [label.size(1)]
        #Perform evaluation of spectrogram
        spec, label = spec.to(device), label.to(device)
        model.add_paddings(spec)
        output = model(spec)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        #Calculate predictions and targets 
        decoded_preds, decoded_targets = GreedyDecoder(output,
            label, label_length, blank_label, int2char)
        #Convert prediction and targets from custom chars to IPA phonemes
        predicted_ipas, target_ipas = chars_to_ipa(decoded_preds,
            decoded_targets, char2ipa)
        
        #Calculate PER and WER
        pers.append(cer(decoded_targets[0], decoded_preds[0]))
        wers.append(wer(decoded_targets[0], decoded_preds[0]))
        #Log PER, WER, target_ipas and predicted_ipas
        log.write(f"{idx+1}\t{pers[-1]:.4f}\t{wers[-1]:.4f}\t{target_ipas[0]}"
                  f"\t\t{predicted_ipas[0]}\n")
        
        if idx%200 == 0:
            print(f"\t{idx+1}/{len(lines)} inferences ran...")

    print(f"Done, all {idx+1} inferences have been run")
    
log.close()

# Log and print average PER and average WER
msg = f"\nI am using this checkpoint {chckpnt_path}\n"
msg += f"I am running inferences for this transcript: {transcr_path}\n"
msg += "\nResults are:\n"
msg += f"Average PER: {(sum(pers)/len(pers)):.4f}\n"
msg += f"Average WER: {(sum(wers)/len(wers)):.4f}"
msg += "\n\nI used the following dictionaries:\n"
for Dict in dicts:
    msg += f"\t{Dict}\n"
    
#Log number of instances found in transcript for each keyword
msg += "\nThis is the number of times each keyword was found in transcript:"
msg += "\n\tWORD\t|\t# of instances\n"
for k_word in list(k_words_num.keys()):
    msg += f"{k_word:>8}\t\t"
    msg += f"{k_words_num[k_word]:>7}\n"

log_message(msg, logfile_path, 'a', True)

'''References:
CTC in Pytorch: https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO#scrollTo=RVJs4Bk8FjjO
Transfer Learning: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
Transfer Learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
Transfer Learning: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
find_lr: Programming PyTorch for Deep Learning'''

'''NOTES:
-Having two consecutive labels together may cause CTC to give an inf loss
-I am removing audio 32-4137-0011 from transcript in LS's train-clean-100
-To create a new "slim" version of SC, use: run_04/SC_full_to_slim.py
-In customdict I switched places of 'one   w ah n' and 'one  hh w ah n'
-I am removing TIMIT's and AOLME's preprocessing functions, since I am
only using SC right now. Refer to folder "run06" to find those functions.
-I am removing SC functions, right now, only spanish (refer to run_08 for SC).
-In run_10, I am changing my early stop logic. Using PER only.'''
