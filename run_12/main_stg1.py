'''/**************************************************************************
    File: main_stg1.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
import copy 
import gc
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

from models import SpeechRecognitionModel
from preprocessing import preprocess_data, check_folder, get_mappings
from utils import data_processing, train, dev, Metrics, log_model_information, \
    plot_and_save, log_message, find_best_lr, BucketsSampler, CUSTOM_DATASET, \
    log_labels, log_k_words_instances

#Comment this from time to time and check warnings are the same
warnings.filterwarnings("ignore")

##############################################################################
#VARIABLES THAT MIGHT NEED TO BE CHANGED ARE ENCLOSED IN THESE HASHTAGS
PREPROCESSING = True #preprocessing, get transcripts ready
FIND_LR = False #find best learning rate
TRAIN = True #train and validate!

desktop_path = str(Path.home()) + '/Desktop/ctc_runs'
#Nomenclature: K=1000; E=epochs
logs_folder = desktop_path + '/SC_14K_50E/stage1'
miscellaneous_log = logs_folder + '/miscellaneous.txt'
train_log = logs_folder + '/train_logs.txt'
checkpoint_path = logs_folder + '/checkpoint.tar'
k_words_path = logs_folder + '/k_words_instances.pickle'
#Used to determine the labels and translations between them
ipa2char, char2ipa, char2int, int2char, blank_label = {}, {}, {}, {}, 0

#PREPROCESSING-----------------------------------------------------------------
data_root = '/media/mario/audios' #root for dictionaries and transcripts
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
    'num': 200 #Set equal to None if you want to use all audios
}

#Kaggle's variables and paths
KA_data = {
    'dataset_ID': 'KA',
    'use_dataset': False,
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
    'use_dataset': True,
    'dict': data_root + '/dict/sc_dict.pickle',
    'src_dir': data_root + '/spctrgrms/clean/SC',
    'train_csv': gt_csvs_folder + '/sc_train.csv',
    'dev_csv': gt_csvs_folder + '/sc_dev.csv',
    'splits': [0.9, 0.1],
    'num': 14000 #Set equal to None if you want to use all audios
}

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
early_stop = {'n': 6, 'p': 0.999}
#TM will be multiplied by the 'time' length of the spectrograms
FM, TM = 27, 0.125 #Frequency and Time Masking Attributes
bucket_boundaries = sorted([2000, 5000, 8000]) #in miliseconds
drop_last = True

GRU = {'dim': [64], 'hid_dim': [64], 'layers': [2]}
#WARNING, if stride is changed, paddings may result incorrect
CNN1 = {'filters': [6], 'kernel': [4], 'stride':[2]}
n_class = -1 #automatically sets up on Step 2
n_mels = [128] #n_feats
dropout = [0.1]
learning_rate = [1e-4]
batch_size = [2]
epochs = [50]

#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#Make sure that assumed-path-to-desktop exists
if not os.path.exists(desktop_path):
    print(f"ERROR: I assumed the path to your desktop was this {desktop_path}"
          ", but it seems I am incorrect. Can you please fix it? Thanks.")
    sys.exit()

#If logs_folder exists, ask if okay to overwrite; otherwise, create it
check_folder(logs_folder)
   
#Get IPA to Char, Char to IPA, Char to Int and Int to Char dictionaries
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings(datasets,
    other_chars, manual_chars)
    
if PREPROCESSING: #------------------------------------------------------------
    #In a nutchell: check audios and create csvs for training
    preprocess_data(gt_csvs_folder, k_words, datasets, train_csv, dev_csv,
                    k_words_path)
    
if TRAIN or FIND_LR: #--------------------------------------------------------
    hyper_params = {"gru_dim": GRU['dim'], "gru_hid_dim": GRU['hid_dim'],
        "gru_layers": GRU['layers'], "cnn1_filters": CNN1['filters'],
        "cnn1_kernel": CNN1['kernel'], "cnn1_stride": CNN1['stride'],
        "n_class": [blank_label+1], "n_mels": n_mels, "dropout": dropout, 
        "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs
    }
    
    num_runs = len(list(ParameterGrid(hyper_params)))
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
    
    #To keep track of best metrics, best model and best hyper params.
    best_cers, global_best_cer, best_model_wts, best_hparams = [], 2.0, {}, {}
    optimizer_state_dict = {}
    
    start_time = time.time()
    for idx, hparams in enumerate(list(ParameterGrid(hyper_params))):
        torch.manual_seed(7)
        random.seed(7)
        msg = f"PARAMETERS [{idx+1}/{num_runs}]\n"
        log_message(msg, train_log, 'a', True)
        msg = f"----------PARAMETERS [{idx+1}/{num_runs}]----------\n"
        log_message(msg, miscellaneous_log, 'a', False)
        
        sampler = BucketsSampler(train_csv, bucket_boundaries, hparams['batch_size'], drop_last)
        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
        #train_loader batch size is set to 1, since we are using BucketsSampler
        train_loader = data.DataLoader(
            dataset=train_dataset, batch_size=1, batch_sampler=sampler,
            collate_fn=lambda x: data_processing(x, char2int, hparams['n_mels'], FM, TM, 'train'),
            **kwargs)
        
        sampler = BucketsSampler(dev_csv, bucket_boundaries, hparams['batch_size'], drop_last)
        #dev_loader batch size is set to 1, since we are using BucketsSampler
        dev_loader = data.DataLoader(dataset=dev_dataset, batch_size=1, batch_sampler=sampler,
            collate_fn=lambda x: data_processing(x, char2int, hparams['n_mels']),
            **kwargs)
                
        model = SpeechRecognitionModel(hparams).to(device)
        optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
        criterion = nn.CTCLoss(blank=blank_label).to(device)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
            steps_per_epoch=train_loader.batch_sampler.num_of_batches,
            epochs=hparams['epochs'], anneal_strategy='linear')
        
        if FIND_LR:
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
                
                stop, stop_msg = metrics.should_we_stop(epoch, early_stop)
                if stop: #Early Stop
                    MSG += stop_msg
                    break
                
            #Get best cer of this run; if best than all runs, copy the model
            local_best_cer = metrics.get_best_cer()
            best_cers.append(local_best_cer)
            if local_best_cer < global_best_cer:
                global_best_cer = local_best_cer
                best_model_wts = copy.deepcopy(model.state_dict())
                optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                best_hparams = hparams
            
            msg = MSG + f"Best CER: {local_best_cer:.4f} on Epoch "
            msg += f"{metrics.cers.index(local_best_cer) + 1}"
            log_message(msg + '\n\n', train_log, 'a', True)
            
            #Log model summary, # of parameters, hyper parameters and more
            log_model_information(miscellaneous_log, model, hparams)
                
            #Plot losses, cers, learning rates and save as figures
            plot_and_save(metrics.dev_losses, metrics.train_losses, metrics.cers,
                metrics.lrs, idx+1, logs_folder)
                
        #Delete model, collect garbage and empty CUDA memory
        #See: https://stackify.com/python-garbage-collection/
        # model.apply(weights_init)
        del model, criterion, scheduler, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    if TRAIN:
        #Save weights of best model, along with its optimizer state and hparams
        torch.save({
            'model_state_dict': best_model_wts,
            # 'optimizer_state_dict': optimizer_state_dict,
            'hparams': best_hparams
        }, checkpoint_path)
        
        #Record "bestest" metrics, run-time, and others
        msg = f"\nBest CER of all was {min(best_cers):.4f} on run "
        msg += f"{best_cers.index(min(best_cers)) + 1}\n"
        msg += f"Checkpoint has been saved here: {checkpoint_path}\n"
        # msg += f"The number of k_words found in dev dataset is {gt_num}\n"
        msg += "Are we using masking during training? 'Yes'\n"
        msg += f"In all runs, training set had {len(train_dataset)} audio files "
        msg += f"equivalent to {train_dataset.duration:.2f} seconds\n"
        msg += f"In all runs, dev set had {len(dev_dataset)} audio files; equi"
        msg += f"valent to {dev_dataset.duration:.2f} seconds\n"
        msg += f"Early Stop Values:\n\tn: {early_stop['n']}\n\tPercentage: "
        msg += f"{((1-early_stop['p'])*100):.2f}%\n"
        msg += f"Number of classes: {hyper_params['n_class']}\n"
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