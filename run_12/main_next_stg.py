'''/**************************************************************************
    File: main_next_stg.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 03/01/2021
    
    TODO
    This stage reads from a previous stage and saves a new (further trained)
    checkpoint.
    
***************************************************************************''' 
import copy 
import gc
import os
from os.path import join as pj #pj stands for path.join
from pathlib import Path
import pickle
import random
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
from utils import data_processing, train, dev, Metrics, CUSTOM_DATASET, \
    plot_and_save, log_message, find_best_lr, BucketsSampler, error, \
    log_model_information, save_chckpnt, log_labels, log_k_words_instances

#Comment this from time to time and check warnings are the same
warnings.filterwarnings("ignore")

##############################################################################
#VARIABLES THAT MIGHT NEED TO BE CHANGED ARE ENCLOSED IN THESE HASHTAGS
PREPROCESSING = True #preprocessing, get transcripts ready
FIND_LR = False #find learning rate
TRAIN = True #train and test!
transf_learn_all_layers = True

#Root location of logs, plots and checkpoints
desktop_path = str(Path.home()) + '/Desktop/ctc_runs'
#Root location of spectrograms and dictionaries
data_root = '/media/mario/audios'

prev_chckpt_dir = 'dummy' #Previous checkpoint folder name
prev_chckpt_stg = 'stg2' #Previous checkpoint stage name
new_chckpt_stg = 'stg3' #Stage name for new checkpoint

#Paths to logs
logs_folder = pj(desktop_path, prev_chckpt_dir, new_chckpt_stg)
miscellaneous_log = pj(logs_folder, 'miscellaneous.txt')
train_log = pj(logs_folder, 'train_logs.txt')

#Paths to checkpoints (old and new)
prev_chckpt_path = pj(desktop_path, prev_chckpt_dir, prev_chckpt_stg,
    'checkpoint_onRun01onEpoch020.tar')
new_chckpt_path = pj(logs_folder, 'checkpoint.tar')

#TODO, implement keywords instances
k_words_path = logs_folder + '/k_words_instances.pickle'

#Used to determine the labels and translations between them
ipa2char, char2ipa, char2int, int2char, blank_label = {}, {}, {}, {}, 0

#PREPROCESSING-----------------------------------------------------------------=
gt_csvs_folder = desktop_path + '/gt'
k_words = ['zero', 'one', 'two', 'three', 'five', 'number', 'numbers', 'cero',
          'uno', 'dos', 'tres', 'cinco', 'número', 'números']
# SR = 16000 # All audios are 16000 due to audios2spctrgrms.py files

#TTS and gTTS's variables and paths (all stored in one dictionary)
TS_data = {
    'dataset_ID': 'TS',
    'use_dataset': False,
    'dict': data_root + '/dict/ts_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TS/transcript.txt',
    'train_csv': gt_csvs_folder + '/ts_train.csv',
    'dev_csv': gt_csvs_folder + '/ts_dev.csv',
    'splits': [0.9, 0.1],
    'num': 1000 #Set equal to None if you want to use all audios
}

#Kaggle's variables and paths
KA_data = {
    'dataset_ID': 'KA',
    'use_dataset': True,
    'dict': data_root + '/dict/ka_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/KA/transcript.txt',
    'train_csv': gt_csvs_folder + '/ka_train.csv',
    'dev_csv': gt_csvs_folder + '/ka_dev.csv',
    'splits': [0.9, 0.1]
}

#TIMIT's variables and paths
TI_data = {
    'dataset_ID': 'TI',
    'use_dataset': False,
    'dict': data_root + '/dict/ti_dict.pickle',
    'transcript': data_root + '/spctrgrms/clean/TI/transcript.txt',
    'train_csv': gt_csvs_folder + '/ti_train.csv',
    'dev_csv': gt_csvs_folder + '/ti_dev.csv',
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
'''Specify which dictionaries should be included that weren't included from
the chosen datasets. For example, if you chose to use KA, the run will only
have 38 classes (number of unique phonemes in KA). But if you want to train
also in english phonemes, you'll have to specify the path or paths to english
dictionary(ies).'''
other_dicts = []

#Specify which datasets you want to use for training
datasets = [TS_data, KA_data, TI_data, SC_data]
#Location of "final" csvs, the ones that will be used to train and validate
train_csv = gt_csvs_folder + '/all_train.csv'
dev_csv = gt_csvs_folder + '/all_dev.csv'

#FIND_LR-----------------------------------------------------------------------
start_lr = 3e-5
max_lr = 1.0

#TRAIN------------------------------------------------------------------------
other_chars = [' '] # other_chars = ["'", ' ']
manual_chars = ['!','?','(',')','+','*','#','$','&','-','=']
early_stop = {'n': 8, 'p': 0.999}
#TM Will be multiplied by the 'time' length of the spectrograms
FM, TM = 27, 0.125 #Frequency and Time Masking Attributes
bucket_boundaries = sorted([2000]) #in miliseconds
drop_last = True

#New config. for dropout? batch_size? epochs? learning_rate?
new_hparams = {'epochs': 1, 'learning_rate': 1e-3}

#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#Make sure that assumed-path-to-desktop exists
if not os.path.exists(desktop_path):
    print(f"{error()} I assumed your desktop's path was {desktop_path}"
          ", but it seems I am incorrect. Can you please fix it? Thanks.")
    sys.exit()
    
if len(other_dicts) == 0:
    print("You are in a Transfer Learning file, are you sure you don't need "
          "other dictionaries? [y/n]")
    if input().lower() != 'y':
        sys.exit()
    
#If logs_folder exists, ask if okay to overwrite; otherwise, create it
check_folder(logs_folder)
    
#Get IPA to Char, Char to IPA, Char to Int and Int to Char dictionaries
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings(datasets,
    other_chars, manual_chars, other_dicts)
    
if PREPROCESSING: #------------------------------------------------------------
    #In a nutchell: check audios and create csvs for training
    preprocess_data(gt_csvs_folder, k_words, datasets, train_csv, dev_csv,
                    k_words_path)
    
if TRAIN or FIND_LR: #--------------------------------------------------------    
    msg = "Training and Validation results are saved here:\n\n"
    log_message(msg, train_log, 'w', False)
    
    msg = "HyperParams, Models Summaries and more will be saved here:\n\n"
    log_message(msg, miscellaneous_log, 'w', False)
    
    #Determine if gpu is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #Create Datasets
    train_dataset = CUSTOM_DATASET(train_csv, ipa2char)
    dev_dataset = CUSTOM_DATASET(dev_csv, ipa2char)
    
    #Initialize seeds
    torch.manual_seed(7)
    random.seed(7)
    
    #Load checkpoint, weights and hyper parameters
    checkpnt = torch.load(prev_chckpt_path)
    hparams = checkpnt['hparams']
    
    #If I want to use new h_params, update checkpoint's hyper parameters
    if bool(new_hparams):
        for k, v in new_hparams.items():
            hparams[k] = v
        
    #Initialize model; 'update' it with checkpoint's parameters
    model = SpeechRecognitionModel(hparams)
    model.load_state_dict(checkpnt['model_state_dict'])
    
    if not transf_learn_all_layers:
        #If using net as "fixed feature extractor" freeze all except last layer
        for param in model.parameters():
            param.requires_grad = False
    
    #Re-initialize last layer
    in_ftrs = model.classifier[3].in_features
    out_ftrs = blank_label+1
    model.classifier[3] = nn.Linear(in_ftrs, out_ftrs)
    model = model.to(device)
    criterion = nn.CTCLoss(blank=blank_label).to(device)
    
    #In order to log correct information, update value for number of classes
    hparams['n_class'] = out_ftrs
    
    if not transf_learn_all_layers:
        #net as "fixed feature extractor" (only optimize the final layer)
        optimizer = optim.AdamW(model.classifier[3].parameters(), hparams['learning_rate'])
    else:
        #"Finetuning the net" (optimize all parameters)
        optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    
    #Set up Train DataLoader
    sampler = BucketsSampler(train_csv, bucket_boundaries, hparams['batch_size'], drop_last)
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    #train_loader batch size is set to 1, since we are using BucketsSampler
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=1, batch_sampler=sampler,
        collate_fn=lambda x: data_processing(x, char2int, hparams['n_mels'], FM, TM, 'train'),
        **kwargs)
    
    #Set up Validation DataLoader
    sampler = BucketsSampler(dev_csv, bucket_boundaries, hparams['batch_size'], drop_last)
    #dev_loader batch size is set to 1, since we are using BucketsSampler
    dev_loader = data.DataLoader(dataset=dev_dataset, batch_size=1, batch_sampler=sampler,
        collate_fn=lambda x: data_processing(x, char2int, hparams['n_mels']),
        **kwargs)
    
    #Set up Learning Rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
        steps_per_epoch=train_loader.batch_sampler.num_of_batches,
        epochs=hparams['epochs'], anneal_strategy='linear')
        
    if FIND_LR:
        find_best_lr(model, criterion, optimizer, train_loader, start_lr,
            max_lr, device)
        
    if TRAIN:
        #To keep track of run time, PERs and best model
        pers, best_per, best_model_wts, epoch_num = [], 2.0, {}, ''
        start_time = time.time()
        
        metrics = Metrics()
        MSG = '\t'
        
        for epoch in range(1, hparams['epochs'] + 1):
            train(model, device, train_loader, criterion, optimizer,
                scheduler, epoch, train_log, blank_label, int2char, char2ipa, 
                metrics)
            
            dev(model, device, dev_loader, criterion, epoch,
                train_log, blank_label, int2char, char2ipa, metrics)
            
            #If current PER is lower than best global PER, copy the model
            epoch_per = metrics.pers[-1]
            if epoch_per < best_per:
                best_per = epoch_per
                best_model_wts = copy.deepcopy(model.state_dict())
                # optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                epoch_num = str(epoch)
            
            stop, stop_msg = metrics.should_we_stop(epoch, early_stop)
            if stop: #Early Stop
                MSG += stop_msg
                break
        
        msg = MSG + f"Best PER: {metrics.get_best_cer():.4f} on Epoch "
        msg += f"{metrics.pers.index(metrics.get_best_cer()) + 1}\n"
        log_message(msg + '\n\n', train_log, 'a', True)
        
        #Log model summary, # of parameters, hyper parameters and more
        log_model_information(miscellaneous_log, model, hparams)
            
        #Plot losses, PERs, learning rates and save as figures
        plot_and_save(metrics.dev_losses, metrics.train_losses, metrics.pers,
            metrics.lrs, 1, logs_folder)
            
        #Save weights of best model, along with its optimizer state and hparams
        new_chckpt_path = save_chckpnt(best_model_wts, hparams, new_chckpt_path,
            '1', epoch_num)
        
        #Log summary of values, paths and attributes used
        msg += f"Checkpoint has been saved here: {new_chckpt_path}\n"
        msg += f"Using all layers in Transf. Learn.? {transf_learn_all_layers}"
        msg += "Are we using masking during training? 'Yes'\n"
        msg += f"In all runs, training set had {len(train_dataset)} audio files "
        msg += f"equivalent to {train_dataset.duration:.2f} seconds\n"
        msg += f"In all runs, dev set had {len(dev_dataset)} audio files; equi"
        msg += f"valent to {dev_dataset.duration:.2f} seconds\n"
        msg += f"Early Stop Values:\n\tn: {early_stop['n']}\n\tPercentage: "
        msg += f"{((1-early_stop['p'])*100):.2f}%\n"
        msg += f"Number of classes: {hparams['n_class']}\n"
        msg += f"Time Masking Coefficient: {TM}, Frequency Masking: {FM}\n"
        msg += f"Buckets Boundaries: {bucket_boundaries}\n"
        msg += f"This run took {(time.time() - start_time):.2f} seconds\n"
        log_message(msg, train_log, 'a', True)
        
        #Log labels' conversions (from IPA to char and char to int)
        log_labels(ipa2char, char2int, miscellaneous_log)
        #Log & print the number of times each k_word appears in train and dev
        log_k_words_instances(k_words_path, miscellaneous_log)
    
        print("\nModels summaries, hyper parameters and other miscellaneous info "
              f"can be found here: {miscellaneous_log}")
        print(f"A log for all trainings can be found here: {train_log}")
        print(f"Losses' Plots for each run can be found here: {logs_folder}")
        
    #Delete model, collect garbage and empty CUDA memory
    #See: https://stackify.com/python-garbage-collection/
    # model.apply(weights_init)
    del model, criterion, scheduler, optimizer
    gc.collect()
    torch.cuda.empty_cache()


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