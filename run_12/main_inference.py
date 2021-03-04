'''/**************************************************************************
    File: main_inference.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 03/03/2021
    
    TODO
    Load checkpoint (model) and perform inferences on specified dataset. 
    Calcuate PER and WER.
    
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
runs_root = str(Path.home()) + '/Desktop/ctc_runs'
data_root = '/media/mario/audios' #path to folder 'data' is

par_dir = 'dummy' #name of run
stg_dir = 'stg1'
chckpnt = 'checkpoint_onRun01onEpoch081.tar'
logfile = 'inferences_on_SC.txt'

transcr_path = data_root + '/spctrgrms/clean/SC/yes/transcript.txt'
dicts = ['/dict/sc_dict.pickle'] #specify dictionaries to use

other_chars = [' ']
manual_chars = ['!','?','(',')','+','*','#','$','&','-','=']
#TODO log information about keyword_instances in transcript
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers', 'cero',
          'uno', 'dos', 'tres', 'cinco', 'número', 'números']
#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#Set up full paths to checkpoint and logfile
chckpnt_path = pj(runs_root, par_dir, stg_dir, chckpnt)
logfile_path = pj(runs_root, par_dir, stg_dir, logfile)

#Add data root to dictionaries' paths
dicts = [data_root + item for item in dicts]

#Make sure directory where logfile is being saved exists
logfile_dir = '/'.join(logfile_path.split('/')[:-1])
if not os.path.exists(logfile_dir):
    print("The folder in which you want to save the inferences' results "
          "doesn't exist. Please double check it.")
    sys.exit()

# Get IPA to Char, Char to IPA, Char to Int and Int to Char dictionaries
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings([],
    other_chars, manual_chars, dicts)

#Determine if gpu is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
    
#Initialize seeds (Imight not need them actually)
torch.manual_seed(7)
random.seed(7)
    
#Load and prep checkpoint for evaluation
checkpnt = torch.load(chckpnt_path)
hparams = checkpnt['hparams']
model = SpeechRecognitionModel(hparams)
model.load_state_dict(checkpnt['model_state_dict'])
model = model.to(device)
model.eval()

#Read lines from transcript that will be used for inferences
transcr = open(transcr_path, 'r')
lines = transcr.readlines()[:50]
transcr.close()

#Iterate through lines, calculate PER and WER for each line
log = open(logfile_path, 'w')
log.write("Line#\tPER\tWER\tTarget IPAs\tPredicted IPAs\n")
words_dict = pickle.load(open(dicts[0], "rb" ))
with torch.no_grad():
    pers, wers = [], []
    for idx, line in enumerate(lines):
        #Get text and spectrogram
        spctrgrm_path, text, _ = line.split('\t')
        spec = torch.load(spctrgrm_path).unsqueeze(1)
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
                  f"\t{predicted_ipas[0]}\n")

log.close()

# Log and print average PER and average WER
msg = f"\nI am using this checkpoint {chckpnt_path}\n"
msg += f"I am running inferences for this transcript: {transcr_path}\n"
msg += "Results are:\n"
msg += f"Average PER: {(sum(pers)/len(pers)):.4f}\n"
msg += f"Average WER: {(sum(wers)/len(wers)):.4f}"
msg += "\n\nI used the following dictionaries:\n"
for Dict in dicts:
    msg += f"\t{Dict}\n"

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