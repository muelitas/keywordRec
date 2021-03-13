'''/**************************************************************************
    File: main_1st_stg.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
import copy 
import gc
import math
import os
from pathlib import Path
import random
from sklearn.model_selection import ParameterGrid
import sys
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import warnings

import constants as cnstnt
from models import SpeechRecognitionModel
from preprocessing import preprocess_data, check_folder, get_mappings
from utils import data_processing, train, dev, Metrics, log_model_information, \
    plot_and_save, log_message, find_best_lr, CUSTOM_DATASET, \
    log_labels, log_k_words_instances, error, save_chckpnt, warn

#TODO implement readablechars2IPA in custom dataset (or data_processing?)
#TODO check that paths exist, particularly the ones for preprocessing
#Comment this from time to time and check warnings are the same
warnings.filterwarnings("ignore")

##############################################################################
#VARIABLES THAT MIGHT NEED TO BE CHANGED ARE ENCLOSED IN THESE HASHTAGS
FIND_LR = False #find best learning rate
TRAIN = True #train and validate!

runs_root = str(Path.home()) + '/Desktop/ctc_runs'
data_root = str(Path.home()) + '/Desktop/ctc_data' #root for dicts and transcripts
#Nomenclature: K=1000; E=epochs
logs_folder = runs_root + '/dummy/stg1'
misc_log = logs_folder + '/miscellaneous.txt'
train_log = logs_folder + '/train_logs.txt'
chckpnt_path = logs_folder + '/checkpoint.tar'
k_words_path = logs_folder + '/k_words_instances.pickle'
#Used to determine the labels and translations between them
ipa2char, char2ipa, char2int, int2char, blank_label = {}, {}, {}, {}, 0

#PREPROCESSING-----------------------------------------------------------------
gt_csvs_folder = runs_root + '/gt'
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers', 'cero',
          'uno', 'dos', 'tres', 'cinco', 'número', 'números']
# SR = 16000 # All audios are 16000 due to audios2spctrgrms.py files

#TTS and gTTS's variables and paths (all stored in one dictionary)
TS_data = {
    'dataset_ID': 'TS',
    'use_dataset': 0,
    'dict': data_root + '/dict/ts_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TS/transcript.txt',
    'train_csv': gt_csvs_folder + '/ts_train.csv',
    'dev_csv': gt_csvs_folder + '/ts_dev.csv',
    'splits': [0.9, 0.1],
    'num': 1000 #Set equal to None if you want to use all audios
}

TSx4 = {
    'dataset_ID': 'TSx4',
    'use_dataset': 1,
    'dict': data_root + '/dict/ts_dict.pickle',
    'transcript': data_root + '/spctrgrms/pyroom/TSx4/transcript.txt',
    'train_csv': gt_csvs_folder + '/ts_x4_train.csv',
    'dev_csv': gt_csvs_folder + '/ts_x4_dev.csv',
    'splits': [0.9, 0.1],
    'num': 1000 #Set equal to None if you want to use all audios
}

TS_kwords = {
    'dataset_ID': 'TS_kw',
    'use_dataset': 0,
    'dict': data_root + '/dict/ts_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TS_kwords/transcript.txt',
    'train_csv': gt_csvs_folder + '/ts_kw_train.csv',
    'dev_csv': gt_csvs_folder + '/ts_kw_dev.csv',
    'splits': [0.9, 0.1],
    'num': None #Set equal to None if you want to use all audios
}

TS_spang = {
    'dataset_ID': 'TS_spang',
    'use_dataset': 0,
    'dict': data_root + '/dict/ts_spang_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TS_spang/transcript.txt',
    'train_csv': gt_csvs_folder + '/ts_spang_train.csv',
    'dev_csv': gt_csvs_folder + '/ts_spang_dev.csv',
    'splits': [0.9, 0.1],
    'num': None #Set equal to None if you want to use all audios
}

#Kaggle's variables and paths
KA_data = {
    'dataset_ID': 'KA',
    'use_dataset': 0,
    'dict': data_root + '/dict/ka_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/KA/transcript.txt',
    'train_csv': gt_csvs_folder + '/ka_train.csv',
    'dev_csv': gt_csvs_folder + '/ka_dev.csv',
    'splits': [0.9, 0.1]
}
#
KAx4 = {
    'dataset_ID': 'KAx4',
    'use_dataset': 0,
    'dict': data_root + '/dict/ka_dict.pickle',
    'transcript': data_root + '/spctrgrms/pyroom/KAx4/transcript.txt',
    'train_csv': gt_csvs_folder + '/ka_x4_train.csv',
    'dev_csv': gt_csvs_folder + '/ka_x4_dev.csv',
    'splits': [0.9, 0.1]
}

#TIMIT's variables and paths
TI_train = {
    'dataset_ID': 'TI_tr',
    'use_dataset': 0,
    'dict': data_root + '/dict/ti_all_train_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TI_all_train/transcript.txt',
    'train_csv': gt_csvs_folder + '/ti_tr_train.csv',
    'dev_csv': gt_csvs_folder + '/ti_tr_dev.csv',
    'splits': [0.9, 0.1]
}
#In case we also want to train with TIMIT's TEST audios
TI_test = {
    'dataset_ID': 'TI_te',
    'use_dataset': 0,
    'dict': data_root + '/dict/ti_all_test_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TI_all_test/transcript.txt',
    'train_csv': gt_csvs_folder + '/ti_te_train.csv',
    'dev_csv': gt_csvs_folder + '/ti_te_dev.csv',
    'splits': [0.9, 0.1]
}

#Speech Commands' variables and paths
SC_data = {
    'dataset_ID': 'SC',
    'use_dataset': False,
    'dict': data_root + '/dict/sc_dict.pickle',
    'src_dir': data_root + '/spctrgrms/clean/SC',
    'train_csv': gt_csvs_folder + '/sc_train.csv',
    'dev_csv': gt_csvs_folder + '/sc_dev.csv',
    'splits': [0.9, 0.1],
    'num': 200 #Set equal to None if you want to use all audios
}

#AOLME's variables and paths
AO_engl = {
    'dataset_ID': 'AO_en',
    'use_dataset': False,
    'dict': data_root + '/dict/ao_en_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/AO_EN/transcript.txt',
    'train_csv': gt_csvs_folder + '/ao_en_train.csv',
    'dev_csv': gt_csvs_folder + '/ao_en_dev.csv',
    'splits': [0.9, 0.1],
    'num': None #Set equal to None if you want to use all audios
}

AO_span = {
    'dataset_ID': 'AO_sp',
    'use_dataset': 0,
    'dict': data_root + '/dict/ao_sp_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/AO_SP/transcript.txt',
    'train_csv': gt_csvs_folder + '/ao_sp_train.csv',
    'dev_csv': gt_csvs_folder + '/ao_sp_dev.csv',
    'splits': [0.9, 0.1],
    'num': None #Set equal to None if you want to use all audios
}

'''Specify which dictionaries should be included that weren't included from
the chosen datasets. For example, if you chose to use KA, the run will only
have 38 classes (number of unique phonemes in KA). But if you want to train
also in english phonemes, you'll have to specify the path or paths to english
dictionary(ies).'''
other_dicts = []

#Specify which datasets you want to use for training
datasets = [TS_data, KA_data, TI_train, TI_test, SC_data, AO_engl, AO_span,
            TS_spang, TS_kwords, TSx4, KAx4]
#Location of "final" csvs, the ones that will be used to train and validate
train_csv = gt_csvs_folder + '/all_train.csv'
dev_csv = gt_csvs_folder + '/all_dev.csv'

#FIND_LR-----------------------------------------------------------------------
start_lr = 3e-5
max_lr = 1.0

#TRAIN------------------------------------------------------------------------
other_chars = [' '] # other_chars = ["'", ' ']
manual_chars = ['!','?','(',')','+','*','#','$','&','-','=',':']
early_stop = {'n': 8, 'p': 0.999, 't': 1.0}
#TM will be multiplied by the 'time' length of the spectrograms
FM, TM = 27, 0.125 #Frequency and Time Masking Attributes

#Hyper Parameters
HP = {  'cnn1_filters': [4],
        'cnn1_kernel': [3],
        'cnn1_stride': [cnstnt.CNN_STRIDE],
        'gru_dim': [32],
        'gru_hid_dim': [32],
        'gru_layers': [2],
        'gru_dropout': [0.1],
        'n_class': [-1], #automatically sets up on Step 2
        'n_mels': [128],
        'dropout': [0.1], #classifier's dropout
        'lr': [3e-4], #learning rate
        'G': [0.96], #Gamma, for learning scheduler
        'bs': [2], #batch size
        'epochs': [2]}

#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#Make sure that assumed-path-to-desktop exists
if not os.path.exists(runs_root):
    print(f"{error()} I assumed your desktop's path was {runs_root}"
          ", but it seems I am incorrect. Can you please fix it? Thanks.")
    sys.exit()

#If logs_folder exists, ask if okay to overwrite; otherwise, create it
check_folder(logs_folder)
   
#Get IPA to Char, Char to IPA, Char to Int and Int to Char dictionaries
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings(datasets,
    other_chars, manual_chars, other_dicts)
    
# PREPROCESSING --------------------------------------------------------------
#In a nutchell: check audios and create csvs for training
random.seed(7)
preprocess_data(gt_csvs_folder, k_words, datasets, train_csv, dev_csv,
                k_words_path, misc_log)
    
if TRAIN or FIND_LR: #--------------------------------------------------------
    HP['n_class'][0] = blank_label+1 #Initialize number of classes (labels)
    
    num_runs = len(list(ParameterGrid(HP)))
    msg = "Training and Validation results are saved here:\n\n"
    log_message(msg, train_log, 'w', False)
    
    msg = "HyperParams, Models Summaries and more will be saved here:\n\n"
    log_message(msg, misc_log, 'a', False)
    
    #Determine if gpu is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #Create Datasets
    train_dataset = CUSTOM_DATASET(train_csv, ipa2char)
    dev_dataset = CUSTOM_DATASET(dev_csv, ipa2char)
    
    #To keep track of best metrics, best model and best hyper params.
    best_pers, global_best_per, best_model_wts, best_hparams = [], 2.0, {}, {}
    optimizer_state_dict, run_num, epoch_num = {}, '', ''
    
    #Iterate through hyper parameters
    start_time = time.time()
    for idx, hparams in enumerate(list(ParameterGrid(HP))):
        torch.manual_seed(7)
        msg = f"PARAMETERS [{idx+1}/{num_runs}]\n"
        log_message(msg, train_log, 'a', True)
        msg = f"----------PARAMETERS [{idx+1}/{num_runs}]----------\n"
        log_message(msg, misc_log, 'a', False)
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}        
        train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=hparams['bs'],
                                    shuffle=True,
                                    collate_fn=lambda x: data_processing(x, char2int),
                                    **kwargs)
        dev_loader = data.DataLoader(dataset=dev_dataset,
                                    batch_size=hparams['bs'],
                                    shuffle=True,
                                    collate_fn=lambda x: data_processing(x, char2int),
                                    **kwargs)
        
        model = SpeechRecognitionModel(hparams).to(device)
        optimizer = optim.AdamW(model.parameters(), hparams['lr'])
        criterion = nn.CTCLoss(blank=blank_label).to(device)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = hparams['G'])
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['lr'], 
        #     steps_per_epoch=math.ceil(len(train_dataset)/hparams['bs']),
        #     epochs=hparams['epochs'], anneal_strategy='linear')
        
        if FIND_LR:
            num_params = sum([param.nelement() for param in model.parameters()])
            print(f"Number of parameters in model is: '{num_params}'")
            find_best_lr(model, criterion, optimizer, train_loader, start_lr,
                max_lr, device)
        
        if TRAIN:
            metrics = Metrics()
            MSG = '\t'
            
            for epoch in range(1, hparams['epochs'] + 1):
                train(model, device, train_loader, criterion, optimizer,
                    scheduler, epoch, train_log, blank_label, int2char, char2ipa, 
                    metrics)
                
                dev(model, device, dev_loader, criterion, epoch,
                    train_log, blank_label, int2char, char2ipa, metrics)
                
                #If current PER is lower than best global PER, copy the model
                epoch_per = metrics.dev_pers[-1]
                if epoch_per < global_best_per:
                    global_best_per = epoch_per
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                    best_hparams = hparams
                    run_num = str(idx+1)
                    epoch_num = str(epoch)
                
                stop, stop_msg = metrics.should_we_stop(epoch, early_stop)
                if stop: #Early Stop
                    MSG += stop_msg
                    break
                
            local_best_per = metrics.get_best_cer()
            best_pers.append(local_best_per)
            
            msg = MSG + f"Best PER: {local_best_per:.4f} on Epoch "
            msg += f"{metrics.dev_pers.index(local_best_per) + 1}"
            log_message(msg + '\n\n', train_log, 'a', True)
            
            #Log model summary, # of parameters, hyper parameters and more
            num_params = log_model_information(misc_log, model, hparams)
                
            #Plot losses, cers, learning rates and save as figures
            plot_and_save(metrics.dev_losses, metrics.train_losses, metrics.dev_pers,
                metrics.train_pers, metrics.lrs, idx+1, logs_folder)
        
        #Delete model, collect garbage and empty CUDA memory
        #See: https://stackify.com/python-garbage-collection/
        # model.apply(weights_init)
        del model, criterion, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    if TRAIN:
        #Save weights of best model, along with its optimizer state and hparams
        chckpnt_path = save_chckpnt(best_model_wts, best_hparams, chckpnt_path,
            run_num, epoch_num)
        
        #Record "bestest" metrics, run-time, and others
        msg = f"\nBest PER of all was {min(best_pers):.4f} on run "
        msg += f"{best_pers.index(min(best_pers)) + 1}\n"
        msg += f"Checkpoint has been saved here: {chckpnt_path}\n"
        msg += f"Number of parameters in model: {num_params}\n"
        msg += "Are we using masking during training? 'Yes'\n"
        msg += f"In all runs, training set had {len(train_dataset)} audio files "
        msg += f"equivalent to {train_dataset.duration:.2f} seconds\n"
        msg += f"In all runs, dev set had {len(dev_dataset)} audio files; equi"
        msg += f"valent to {dev_dataset.duration:.2f} seconds\n"
        msg += f"Early Stop Values:\n\tn: {early_stop['n']}\n\tPercentage: "
        msg += f"{((1-early_stop['p'])*100):.2f}%\n\tOverfit Threshold: "
        msg += f"{early_stop['t']:.2f}\n"
        msg += f"Number of classes: {HP['n_class']}\n"
        msg += f"Time Masking Coefficient: {TM}, Frequency Masking: {FM}\n"
        msg += f"This run took {(time.time() - start_time):.2f} seconds\n"
        log_message(msg, train_log, 'a', True)
        
        #Log labels' conversions (from IPA to char and char to int)
        log_labels(ipa2char, char2int, misc_log)
        #Log & print the number of times each k_word appears in train and dev
        log_k_words_instances(k_words_path, misc_log)
    
        print("\nModels summaries, hyper parameters and other miscellaneous info "
              f"can be found here: {misc_log}")
        print(f"A log for all trainings can be found here: {train_log}")
        print(f"Losses' Plots for each run can be found here: {logs_folder}")

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
-In run_10, I am changing my early stop logic. Using CER only.
-In run_11, I am loading saved spectrograms instead of calculating them.
-In run 12, we migrate to github.'''
