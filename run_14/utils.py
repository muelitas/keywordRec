'''/**************************************************************************
    File: utils.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
#Source: https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO#scrollTo=RVJs4Bk8FjjO
#Batch Sampler Idea from: https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio import transforms as tforms

import constants as cnstnt
from models import BiGRU

def plot_spctrgrm(title, spctrgrm):
    '''Plot spctrgrm with specified {title}'''
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    plt.imshow(spctrgrm.log2()[0,:,:].detach().numpy(), cmap='viridis')
    plt.show()    

def log_model_information(log_file, model, hparams):
    '''Log model summary, # of parameters, hyper parameters and more'''
    original_stdout = sys.stdout # Save a reference to the original std output
    with open(log_file, 'a') as log:
        num_params = sum([param.nelement() for param in model.parameters()])
        sys.stdout = log # Change standard output to the created file
        #Number of parameters
        print('Total Number of Parameters in the Model is: ',end='')
        print(num_params)
        
        #Model Summary
        print("\nModel's Summary:")
        print(model) #print model summary in {log_file}
        
        #Hyper Parameters
        print('\nHyper Parameters:')
        for k,v in hparams.items():
            print(f'\t{k}: {v}')
        
        print('\n')
        #Reset the std output to its original value (the terminal)
        sys.stdout = original_stdout 
        
        return num_params

def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)
    
    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')
        
def log_labels(ipa2char, char2int, log_path):
    '''Log the conversions that took place from IPA to char and from char to
    int. In other words, log which integer (label) represented each phoneme'''
    msg = '\nThe label representation for all runs was the following:\n'
    msg += '   IPA char  int\n'
    for [k1, v1], [_, v2] in zip(ipa2char.items(), char2int.items()):
        msg += f"{k1:>5}{v1:>5}{v2:>5}\n"
    
    msg += f"'Blank Label': {len(ipa2char.items())}\n"
    log_message(msg, log_path, 'a', False)
    
def log_k_words_instances(k_words_path, log_path):
    '''Log the number of times each k_word was found in train and dev csvs'''
    #Get instances from saved pickle file
    k_words_num = pickle.load(open(k_words_path, "rb" ))
    
    #Log them in {log_path}
    msg = "\nThis is the number of times each keyword was found in csvs:"
    msg += "\n\tWORD\t|\tInstances in TRAIN\t|\tInstances in VALIDATION\n"
    for k_word in list(k_words_num['train'].keys()):
        msg += f"{k_word:>12}\t\t"
        msg += f"{k_words_num['train'][k_word]:>10}\t\t\t"
        msg += f"{k_words_num['dev'][k_word]:>13}\n"
    
    log_message(msg, log_path, 'a', False)

def delete_contents(this_dir):
    """Delete contents from folder located in {this_dir}"""
    for item in os.listdir(this_dir):
        if os.path.isfile(this_dir + '/' + item):
            os.remove(this_dir + '/' + item)
        elif os.path.isdir(this_dir + '/' + item):
            shutil.rmtree(this_dir + '/' + item)
        else:
            print(f"I couln't remove this file {this_dir + '/' + item}")
            print("Please try manually. Then, re-run this code.\n")
            sys.exit()

def plot_and_save(dev_losses, train_losses, dev_pers, train_pers, lrs, run,
                  logs_folder):
    '''In figure one, plot and save dev_losses vs train_losses. In figure two,
    plot and save validation PERs vs train PERs. In figure three, plot and
    save progress of learning rate.'''
    file_name = f'/zRun_{str(run).zfill(3)}'
    
    #Validation and Training Losses
    fig_name = logs_folder + file_name + '.png'
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(f'Run {run}: Valid Loss vs. Train Loss')
    x = list(range(1, len(dev_losses)+1))
    ax.plot(x, dev_losses, 'b', label="Validation Loss")
    ax.plot(x, train_losses, 'r', label="Train Loss")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.legend(loc='upper center', shadow=True, fontsize='small')
    plt.savefig(fig_name)
    plt.show()
    
    #Validation Phoneme Error Rates
    fig_name = logs_folder + file_name + '_PER.png'
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(f'Run {run}: Phoneme Error Rates (PERs)')
    x = list(range(1, len(dev_pers)+1))
    ax.plot(x, dev_pers, 'b', label="Validation PERs")
    ax.plot(x, train_pers, 'r', label="Train PERs")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PERs')
    ax.legend(loc='upper center', shadow=True, fontsize='small')
    plt.savefig(fig_name)
    plt.show()
    
    #Learning Rate Progress
    fig_name = logs_folder + file_name + '_LR.png'
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(f'Run {run}: Learning Rate Progress')
    x = list(range(1, len(lrs)+1))
    ax.plot(x, lrs, 'g--', label="Learning Rate")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rates')
    ax.legend(loc='upper center', shadow=True, fontsize='small')
    plt.savefig(fig_name)
    plt.show()
    
class Metrics:
    """TODO
    """
    def __init__(self):
        '''
        TODO
        '''
        self.train_losses = []
        self.train_pers = []
        self.lrs = []
        self.dev_losses = []
        self.dev_pers = [] #Phoneme error rates
        self.ratio_losses = []
        
    def add_train_metrics(self, loss, per, lr):
        '''Add train loss, train PER and Learning Rate'''   
        self.train_losses.append(loss)
        self.train_pers.append(per)
        self.lrs.append(lr)
        
    def add_dev_metrics(self, loss, per, ratio_loss):
        '''Add dev loss, ratio loss and PER'''   
        self.ratio_losses.append(ratio_loss)
        self.dev_losses.append(loss)
        self.dev_pers.append(per)
        
    def get_best_cer(self):
        '''Grab best PER'''
        return min(self.dev_pers)
    
    def should_we_stop(self, epoch, early_stop):
        """If PER doesn't improve by %p in n epochs, stop training; where 
        n = early_stop['n'] and p = (1-early_stop['p'])*100. On the other
        hand, if we have overfitting above t for n epochs, stop training.
        Where t = early_stop['t']. Start checking for early stop once w epochs
        have passed; where w = early_stop['w']"""
        stop, msg = False, ''
        
        if(epoch >= early_stop['w']):
            prev_pers = self.dev_pers[-early_stop['n']:]
            if(prev_pers[0] * early_stop['p'] - min(prev_pers[1:]) < 0.00001):
                stop = True
                msg = 'EARLY STOP due to PER | '
            
            #n previous ratio losses
            ratio_losses = self.ratio_losses[-early_stop['n']:]
            #If all of them are above threshold t, stop due to overfitting
            counter = [1 if i > early_stop['t'] else 0 for i in ratio_losses]
            if sum(counter) == early_stop['n']:
                stop = True
                msg = 'EARLY STOP due to OVERFIT | '
                
        return stop, msg
            
def chars_to_int(text, char2int):
    '''Convert string of chars to list of integers'''
    integers = []
    text = text.replace(' ', '_ _')
    chars = text.split('_')
    for ch in chars:
        integers.append(char2int[ch])
                
    return integers

def int_to_chars(ints_list, int2char):
    '''Convert list of ints to string of custom characters'''
    #THIS was the old way
    # text = []
    # for i in ints_list:
    #     text.append(int2char[i])
        
    # text = '_'.join(text)
    # text = text.replace('_ _', ' ')
    
    text = ''
    for i in ints_list:
        text += int2char[i]
    
    return text

def chars_to_ipa(predictions, targets, char2ipa):
    '''Trsanslate decoded predictions and decoded targets to ipa phonemes.
    Phonemes are separated by an underscore.'''
    predicted_ipas, target_ipas = [], []
    
    for prediction in predictions:
        new_words = []
        for word in prediction.split(' '):
            new_word = []
            for ch in word:
                if ch == '':
                    new_word.append('')
                else:
                    new_word.append(char2ipa[ch])
                
            new_words.append('_'.join(new_word))
            
        predicted_ipas.append(' '.join(new_words))
        
    for target in targets:
        new_words = []
        for word in target.split(' '):
            new_word = []
            for ch in word:
                if ch == '':
                    new_word.append('')
                else:
                    new_word.append(char2ipa[ch])
                
            new_words.append('_'.join(new_word))
            
        target_ipas.append(' '.join(new_words))   
        
    return predicted_ipas, target_ipas

def readablechars2IPA(text, words_dict):
    '''Used for inferencing. Convert readable characters in {text} to IPA
    phonemes using translations in {words_dict}'''
    new_words = []
    words = text.split(' ')
    for word in words:
        phs_seq = words_dict[word] #phonemes translation of word
        phs_seq = phs_seq.replace(' ', '_')
        new_words.append(phs_seq)
    
    text = ' '.join(new_words)
    return text

def IPA2customchars(text, ipa2char):
    '''Convert IPA phonemes in {text} to custom characters using the mapping
    in {ipa2char}. Used for inferences.'''
    words = text.split(' ')
    new_words = []
    for word in words:
        new_word = []
        ipa_phones = word.split('_')
        for ph in ipa_phones:
            new_word.append(ipa2char[ph])
            
        new_words.append('_'.join(new_word))
        
    text = ' '.join(new_words)
    return text

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def data_processing(data, char2int, FM=27, TM=0.125, CASE=False):
    spectrograms, labels, inp_lengths, label_lengths, filenames = [],[],[],[],[]
    
    for (spctrgrm_path, utterance) in data:   
        spec = torch.load(spctrgrm_path)
        #Apply audio transforms (frequency and time masking) to train samples
        if CASE:
            spec = tforms.FrequencyMasking(FM)(spec)
            spec = tforms.TimeMasking(int(TM * spec.shape[2]))(spec)
                                
        spec = spec.squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(chars_to_int(utterance, char2int))
        labels.append(label)
        inp_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
        filenames.append('/'.join(spctrgrm_path.strip().split('/')[-2:]))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return spectrograms, labels, inp_lengths, label_lengths, filenames

def GreedyDecoder(output, labels, label_lengths, blank_label, int2char, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(int_to_chars(labels[i][:label_lengths[i]].tolist(), int2char))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(int_to_chars(decode, int2char))
	return decodes, targets


def train(model, device, train_loader, criterion, optimizer, scheduler,
          epoch, log_file, blank_label, int2char, char2ipa, losses):
    model.train()
    msg = f"\tEpoch: {epoch} | "
    
    train_losses, lrs, train_per = [], [], []
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths, filenames = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        
        model.add_paddings(spectrograms)
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        train_losses.append(loss.detach().item())
        loss.backward()
        
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1),
                labels, label_lengths, blank_label, int2char)
        
        for j in range(len(decoded_preds)):
                train_per.append(cer(decoded_targets[j], decoded_preds[j]))
       
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
    
    avg_per = sum(train_per)/len(train_per)        
    train_loss = sum(train_losses) / len(train_losses) #epoch's average loss
    avg_lr = sum(lrs) / len(lrs)
    losses.add_train_metrics(train_loss, avg_per, avg_lr)
    
    msg += f"Train Avg Loss: {train_loss:.4f}\n"
    log_message(msg, log_file, 'a', True)
    #update learning rate
    new_lr = scheduler.step() 
    optimizer.param_groups[0]['lr'] = new_lr
            
def dev(model, device, dev_loader, criterion, epoch, log_file, blank_label,
        int2char, char2ipa, metrics):
    
    batches_num = math.ceil(dev_loader.batch_sampler.sampler.num_samples / 
                            dev_loader.batch_size)
    #used to print decoded predictions and targets
    step = batches_num//2 + 1
    
    model.eval()
    dev_losses, dev_per, dev_wer = [], [], []
    decoded_preds, decoded_targets, MSG = [], [], ''
    
    msg = f"\tEpoch: {epoch} | "
    with torch.no_grad():
        for i, _data in enumerate(dev_loader):
            spectrograms, labels, input_lengths, label_lengths, _ = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            model.add_paddings(spectrograms)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            dev_losses.append(loss.detach().item())

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1),
                labels, label_lengths, blank_label, int2char)
            
            #Decode prediction and targets into ipa phonemes (to assess progress)
            predicted_ipas, target_ipas = chars_to_ipa(decoded_preds,
                decoded_targets, char2ipa)
                                
            #Log some more predictions
            if i % step == 0:
                for k in range(0,len(spectrograms)):
                    MSG += f'\t\tBatch [{i+1}-{k}/{batches_num}] | '
                    MSG += f"{target_ipas[k]}' -> '{predicted_ipas[k]}'\n"
            
            for j in range(len(decoded_preds)):
                dev_per.append(cer(decoded_targets[j], decoded_preds[j]))
                dev_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_per = sum(dev_per)/len(dev_per)
    avg_wer = sum(dev_wer)/len(dev_wer)
    dev_loss = sum(dev_losses) / len(dev_losses) #epoch's average loss
    ratio_loss = dev_loss / metrics.train_losses[epoch-1]
    msg += f"Dev Avg Loss: {dev_loss:.4f} | Ratio Loss: {ratio_loss:.4f} | "
    msg += f"Avg PER: {avg_per:.4f} | Avg WER: {avg_wer:.4f}\n"
    
    log_message(msg, log_file, 'a', True)
    log_message(MSG, log_file, 'a', True)
    metrics.add_dev_metrics(dev_loss, avg_per, ratio_loss)

class CUSTOM_DATASET(Dataset):
    """
    TODO
    """
    def __init__(self, csv_file, ipa2char):
        #Initialize lists that will hold paths to audios, texts and durations
        self.spctrgrms_paths = []
        self.texts = []
        self.duration = 0
        
        #Iterate through csv file and get spctrgrms_paths and texts
        f = open(csv_file, 'r')
        lines = f.readlines()
        for line in lines:
            spctrgrm_path, text, duration = line.strip().split('\t')
            self.spctrgrms_paths.append(spctrgrm_path)
            
            #Convert IPA phonemes in csv to custom characters
            words = text.split(' ')
            new_words = []
            for word in words:
                new_word = []
                ipa_phones = word.split('_')
                for ph in ipa_phones:
                    new_word.append(ipa2char[ph])
                    
                new_words.append('_'.join(new_word))
                
            self.texts.append(' '.join(new_words))
            self.duration += float(duration)/1000 #to convert to seconds
                        
        f.close() 

    def __getitem__(self, n):
        """
        TODO
        """
        spctrgrm_path = self.spctrgrms_paths[n]
        text = self.texts[n]
        return spctrgrm_path, text

    def __len__(self):
        return len(self.spctrgrms_paths)
    
def error():
    '''Prints 'ERROR' in red'''
    return cnstnt.R + "ERROR:" + cnstnt.W

def warn():
    '''Prints 'WARNING' in orange'''
    return cnstnt.P + "WARNING:" + cnstnt.W

def save_chckpnt(best_model_wts, best_hparams, checkpoint_path, run_num,
                     epoch_num):
    '''Save the best models's weights and hyper parameters used for such'''
    save_path = checkpoint_path[:-4] + f'_onRun{run_num.zfill(2)}'
    save_path += f'onEpoch{epoch_num.zfill(3)}' + checkpoint_path[-4:]
        
    #This is the best model of all epochs and all runs
    torch.save({
        'model_state_dict': best_model_wts,
        # 'optimizer_state_dict': optimizer_state_dict,
        'hparams': best_hparams
    }, save_path)
    
    return save_path

def update_cnn(CNN, GRU, model, hparams, oparams):
    """If specified, edit the configuration of our models' CNN layer. If not,
    freeze its parameters."""
    if CNN['change']: #If CNN is being changed:
        #Modify CNN; append parameters to optimize
        model.cnn = nn.Conv2d(1, CNN['cnn1_filters'], CNN['cnn1_kernel'], 
            stride=CNN['cnn1_stride'])
        oparams += model.cnn.parameters()
        #If I modify CNN, I have to modify fc layer that comes next
        in_feats = CNN['cnn1_filters'] * (hparams['n_mels']//CNN['cnn1_stride'])
        #Keep GRU dimensions used in checkpoint
        model.fully_connected = nn.Linear(in_feats, hparams['gru_dim'])
        #If GRU isn't changed, append parameters to optimize here
        if not GRU['change']:
            oparams += model.fully_connected.parameters()
            #Otherwise, do it in GRU function
        
        #Update {hparams} variable with values in {CNN} variable
        for k, v in CNN.items():
            if k in hparams.keys():
                hparams[k] = v
                
        print("\tDone...CNN layer modified; 'hparams' updated")
    
    else: #If it is not being changed, freeze its parameters
        # for name, param in model.named_parameters():
        for idx, param in enumerate(model.cnn.parameters()):
            param.requires_grad = False
        
        print(f'\tDone...Froze {idx+1} parameters in CNN layer(s)')

    return model, hparams, oparams

def update_bigru(CNN, GRU, model, hparams, oparams):
    """If specified, edit the configuration of our models' GRU layers. If not,
    freeze their parameters."""
    if GRU['change']: #If Bi-GRU is being changed:
        #First, modify output features of cnn's fully connected layer
        in_feats = model.fully_connected.in_features
        model.fully_connected = nn.Linear(in_feats, GRU['gru_dim'])
        #Append to list of parameters to optimize
        oparams += model.fully_connected.parameters()
        #Then, modify bidirectional recurrent neural nets
        model.birnn_layers = BiGRU(GRU['gru_dim'], GRU['gru_hid_dim'],
            GRU['gru_layers'], GRU['gru_dropout'], batch_first=True)
        oparams += model.birnn_layers.parameters()
        #Update {hparams} variable with values in {GRU} variable
        for k, v in GRU.items():
            if k in hparams.keys():
                hparams[k] = v
        
        print("\tDone...CNN's Fully Conn. Layer modified")
        print("\tDone...GRU layers modified; 'hparams' updated")
    
    else: #If it is not being changed, freeze its parameters
        #If CNN didn't change, we update first fully connected layer
        if not CNN['change']:
            for idx, param in enumerate(model.fully_connected.parameters()):
                param.requires_grad = False
                
            print(f"\tDone...Froze {idx+1} params in CNN's Fully Conn. Layer")
                
        #Freeze 'birnn' parameters
        for idx, param in enumerate(model.birnn_layers.parameters()):
            param.requires_grad = False
        
        print(f'\tDone...Froze {idx+1} parameters in GRU layer(s)')

    return model, hparams, oparams

def update_classifier(GRU, model, hparams, oparams):
    """If specified, edit the configuration of our models' classifier. If not,
    freeze parameters, except for the very last layer."""
    fc1_out_ftrs = model.classifier[3].in_features #same as fc2_in_ftrs
    
    #If GRU changed, update first fully connected layer
    if GRU['change']:
        fc1_in_ftrs = hparams['gru_hid_dim']*2
        fc1_out_ftrs = 0
        if fc1_in_ftrs <= hparams['n_class']:
            fc1_out_ftrs = hparams['n_class']
        else:
            fc1_out_ftrs = (fc1_in_ftrs - hparams['n_class']) // 2
            fc1_out_ftrs += hparams['n_class']
            
        model.classifier[0] = nn.Linear(fc1_in_ftrs, fc1_out_ftrs)
        oparams += model.classifier[0].parameters()
        print("\tDone...Fully Connected Layer 1 modified")
        
    else: #if it didn't, freeze the fully connected layer
        for idx, param in enumerate(model.classifier[0].parameters()):
            param.requires_grad = False
        
        print(f'\tDone...Froze {idx+1} parameters in Fully Conn. Layer 1')
    
    #If classifier's dropout changed, update it
    if model.classifier[2].p != hparams['dropout']:
        model.classifier[2].p = hparams['dropout']
        print("\tDone...Classifier's dropout has been updated")
        
    if fc1_out_ftrs < hparams['n_class']:
        print(f"\t{warn()} Number of input features in last fully conn. layer"
              f" ({fc1_out_ftrs}) is smaller than number of classes "
              f"({hparams['n_class']})")
        
    #No matter what, we want the last fully conn. layer to be updated
    model.classifier[3] = nn.Linear(fc1_out_ftrs, hparams['n_class'])
    oparams += model.classifier[3].parameters()
    print("\tDone...Last Layer modified")
    return model, oparams

class LR_SCHED():
    """Learning Rate Scheduler; implementation from Deep Learning Book,
    Chapter 8"""
    def __init__(self, params):
        #Initialize starting learning rate, T coeff and ending learning rate
        self.e_0 = params['e_0'] #initial learning rate
        self.e_T = params['e_0'] * 0.01 #max/final learning rate
        self.T = params['T']
        self.k = 1 #counter

    def step(self):
        """Determine and return new learning rate"""
        if self.T == -1:
            #In case I want to implement a 'steady' learning rate
            e_k = self.e_0
        else:
            if self.k <= self.T: 
                alpha = self.k / self.T
                e_k = (1 - alpha) * self.e_0 + alpha * self.e_T
                self.k += 1 #Update counter
            else:
                e_k = self.e_T
           
        return e_k