import copy
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import warnings

import constants as cnstnt
from utils import check_folder, get_mappings, SC_DATASET
from utils import log_message, data_processing, dev
from utils import SpeechRecognitionModel, log_model_information, train, Metrics
from utils import plot_and_save, log_labels, save_chckpnt, LR_SCHED

#Comment this from time to time and check warnings are the same
warnings.filterwarnings("ignore")

#Initialize device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Initialize Parameters and Paths
TRAIN = 1

#Root location of logs, plots and checkpoints
runs_root = str(Path.home()) + '/Desktop/ctc_runs'
#Root location of spectrograms and lists
data_root = '/media/mario/audios/sc_v2_redownload_spctrgrms'

logs_folder = runs_root + '/SC_40E'
misc_log = logs_folder + '/miscellaneous.txt'
train_log = logs_folder + '/train_logs.txt'
chckpnt_path = logs_folder + '/checkpoint.tar'
gt_csvs_folder = runs_root + '/gt'

#Used to determine the labels and translations between them
ipa2char, char2ipa, char2int, int2char, blank_label = {}, {}, {}, {}, 0

#Speech Commands' variables and paths
SC = {
    'dict': '/home/mario/Desktop/ctc_data/dict/sc_dict.pickle',
    'src_dir': data_root,
    'dev_txt': data_root + '/validation_list.txt',
    'test_txt': data_root + '/testing_list.txt',
    'train_txt': data_root + '/training_list.txt'
}

#TRAIN------------------------------------------------------------------------
seed = 7
early_stop = {'n': 6, 'p': 0.999, 't': 1.0, 'w': 8}
#TM will be multiplied by the 'time' length of the spectrograms
FM, TM = 27, 0.125 #Frequency and Time Masking Attributes
specAug = True #Whether to use spec augment during training

#Hyper Parameters
HP = {  'cnn1_filters': 4,
        'cnn1_kernel': 3,
        'cnn1_stride': cnstnt.CNN_STRIDE,
        'gru_dim': 64,
        'gru_hid_dim': 64,
        'gru_layers': 2,
        'gru_dropout': 0.1,
        'n_class': -1, #automatically sets up on Step 2
        'n_mels': 128,
        'dropout': 0.1, #classifier's dropout
        'e_0': 6e-4, #initial learning rate
        'T': 30, #Set to -1 if you want a steady LR throughout training
        'bs': 2, #batch size
        'epochs': 40}


#YOU SHOULDN'T HAVE TO EDIT ANY VARIABLES FROM HERE ON
##############################################################################
#If directories exists, ask if okay to overwrite; otherwise, create them
check_folder(logs_folder)

#Get IPA to Char, Char to IPA, Char to Int and Int to Char dictionaries
ipa2char, char2ipa, int2char, char2int, blank_label = get_mappings(SC['dict'])

if TRAIN: #--------------------------------------------------------
    HP['n_class'] = blank_label+1 #Initialize number of classes (labels)
    
    msg = "Training and Validation results are saved here:\n\n"
    log_message(msg, train_log, 'w', False)
    msg = "HyperParams, Models Summaries and more are saved here:\n\n"
    log_message(msg, misc_log, 'a', False)
    
    #Create Datasets
    dev_dataset = SC_DATASET(SC['dev_txt'], ipa2char, SC)
    test_dataset = SC_DATASET(SC['test_txt'], ipa2char, SC)
    train_dataset = SC_DATASET(SC['train_txt'], ipa2char, SC)

    #To keep track of best metrics, best model and best hyper params.
    best_per, best_model_wts = 2.0, {}
    optimizer_state_dict, run_num, epoch_num = {}, '-1', '-1'

    #Set up seeds (for randomness) and start timer
    start_time = time.process_time()
    torch.manual_seed(seed)
    random.seed(seed)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}        
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=HP['bs'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, char2int, FM, TM, specAug),
                                **kwargs)
    dev_loader = data.DataLoader(dataset=dev_dataset,
                                batch_size=HP['bs'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, char2int),
                                **kwargs)
    
    model = SpeechRecognitionModel(HP).to(device)
    optimizer = optim.AdamW(model.parameters(), HP['e_0'])
    criterion = nn.CTCLoss(blank=blank_label).to(device)
    scheduler = LR_SCHED(HP)
        
    metrics = Metrics()
    MSG = '\t'
    
    for epoch in range(1, HP['epochs'] + 1):
        train(model, device, train_loader, criterion, optimizer,
            scheduler, epoch, train_log, blank_label, int2char, char2ipa, 
            metrics)
        
        dev(model, device, dev_loader, criterion, epoch,
            train_log, blank_label, int2char, char2ipa, metrics)
        
        #Check if ratio loss is between 1.01 and 0.99
        keep_it = metrics.keep_result()
        
        #If current PER is lower than best PER, copy the model
        epoch_per = metrics.dev_pers[-1]
        if epoch_per < best_per and keep_it:
            best_per = epoch_per
            best_model_wts = copy.deepcopy(model.state_dict())
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            epoch_num = str(epoch)
        
        stop, stop_msg = metrics.should_we_stop(epoch, early_stop)
        if stop: #Early Stop
            MSG += stop_msg
            break

    #TODO check if best_per equals metrics.get_best_cer()
    #TODO check if epoch_num equals metrics.dev_pers.index(metrics.get_best_cer())
    msg = MSG + f"Best PER: {metrics.get_best_cer():.4f} on Epoch "
    msg += f"{metrics.dev_pers.index(metrics.get_best_cer()) + 1}"
    log_message(msg + '\n\n', train_log, 'a', True)
    
    #Log model summary, # of parameters, hyper parameters and more
    num_params = log_model_information(misc_log, model, HP)
        
    #Plot losses, cers, learning rates and save as figures
    plot_and_save(metrics.dev_losses, metrics.train_losses, metrics.dev_pers,
        metrics.train_pers, metrics.lrs, 1, logs_folder)
    
    #Save weights of best model, along with its optimizer state and HP
    chckpnt_path = save_chckpnt(best_model_wts, HP, chckpnt_path,
        '1', epoch_num, optimizer_state_dict, train_log)
    
    #Record "bestest" metrics, run-time, and others
    msg += f"Checkpoint has been saved here: {chckpnt_path}\n"
    msg += f"Number of parameters in model: {num_params}\n"
    msg += f"In all runs, training set had {len(train_dataset)} audio files\n"
    msg += f"In all runs, dev set had {len(dev_dataset)} audio files\n"
    msg += f"Early Stop Values:\n\tn: {early_stop['n']}\n\tPercentage: "
    msg += f"{((1-early_stop['p'])*100):.2f}%\n\tOverfit Threshold: "
    msg += f"{early_stop['t']:.2f}\n\tNumber of epochs to wait: "
    msg += f"{early_stop['w']}\n"
    msg += f"Number of classes: {HP['n_class']}\n"
    msg += f"Are we using masking during training? {specAug}\n"
    if specAug:
        msg += f"Time Masking Coeff.: {TM}, Frequency Masking: {FM}\n"
    msg += f"This run took {(time.process_time() - start_time):.2f} seconds\n"
    log_message(msg, train_log, 'a', True)
    
    #Log labels' conversions (from IPA to char and char to int)
    log_labels(ipa2char, char2int, misc_log)

    print("\nModels summaries, hyper parameters and other miscellaneous info "
            f"can be found here: {misc_log}")
    print(f"A log for all trainings can be found here: {train_log}")
    print(f"Losses' Plots for each run can be found here: {logs_folder}")

