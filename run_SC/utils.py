'''/**************************************************************************
    File: preprocessing.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
from glob import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from random import randint
import shutil
import string
import sys
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms as tforms
from typing import Tuple

import constants as cnstnt

def error():
    '''Prints 'ERROR' in red'''
    return cnstnt.R + "ERROR:" + cnstnt.W

def warn():
    '''Prints 'WARNING' in orange'''
    return cnstnt.P + "WARNING:" + cnstnt.W

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

def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)
    
    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')

def check_folder(this_path):
    '''If {this_path} exists, ask if okay to overwrite; otherwise create it'''
    #Get parent folder and make sure it exists
    par_dir = '/'.join(this_path.split('/')[:-1])
    if not os.path.isdir(par_dir):
            os.mkdir(par_dir)   
            print(f"Parent folder '{par_dir.split('/')[-1]}' has been created")
    
    #Now, make sure {this_path} exists. If it does, ask if okay to overwrite.
    if not os.path.isdir(this_path):
            os.mkdir(this_path)    
    
    if len(os.listdir(this_path)) != 0:
        print(f"{this_path} isn't empty, is it okay if I overwrite it?")
        if input() != 'y':
            sys.exit()
        else:
            delete_contents(this_path)

def ask_user(this_msg):
    """
    TODO
    """
    print("WARNING: " + this_msg)
    answer = input("What do you want me to do? 'c' continue or 'q' quit ")
    if answer.lower() == 'q':
        print("You got it. Bye.")
        sys.exit()
    elif answer.lower() == 'c':
        print("Okay, just know that you might get error(s) later on")
        return
    else:
        print(f"I don't know this '{answer}' command. To be safe, bye.")
        sys.exit()

def get_custom_chars():
    '''Get list of custom characters that will replace IPA phonemes'''
    chars = []
    for c in string.ascii_lowercase:
        chars.append(c)
        
    for c in string.ascii_uppercase:
        chars.append(c)
        
    for c in range(0,10):
        chars.append(str(c))
    
    return chars

def get_mappings(Dict):
    '''Returns IPA to Char, Char to IPA, Char to Int and Int to Char 
    dictionaries'''
    # Get a unique list of IPA phonemes present in our dictionaries. Only from
    # those dictionaries that we wish to use
    IPA_phones = []
    words_in_dict = pickle.load(open(Dict, "rb" ))
    for phonemes in list(words_in_dict.values()):
        for ph in phonemes.split(' '):
            if ph not in IPA_phones:
                IPA_phones.append(ph)
    
    IPA_phones = sorted(IPA_phones)
    
    #Get list of characters that will replace IPA phonemes in csvs
    custom_chars = get_custom_chars()
    
    #Check that I have enough custom characters
    if len(IPA_phones) > len(custom_chars):
        ask_user("You don't have enough custom characters :( ")
        
    #Create ipa-to-character and character-to-ipa maps (dictionaries)
    ipa2char, char2ipa = {}, {}
    for ph, ch in zip(IPA_phones, custom_chars):
        ipa2char[ph] = ch
        char2ipa[ch] = ph
        
    #Create int-to-char and char-to-int maps (dictionaries)
    int2char, char2int = {}, {}
    for idx, ch in enumerate(list(char2ipa.keys())):
        int2char[idx] = ch
        char2int[ch] = idx
            
    return ipa2char, char2ipa, int2char, char2int, idx+1
        
def create_txt_files(SC, misc_log, seed):
    '''Create custom txt files with two columns: full path to drive and word
    said in each audio as a sequence of IPA phonemes'''
    print("Processing Speech Commands' Dataset...")
    random.seed(seed)
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(SC['dict'], "rb" ))
    
    #Read lines from old dev .txt
    old_dev = open(SC['old_dev_txt'], 'r')
    dev_lines = old_dev.readlines()
    old_dev.close()
    random.shuffle(dev_lines)
    
    #Create new dev .txt (add src_dir and append phonemes-translation)
    new_dev = open(SC['new_dev_txt'], 'w')
    for line in dev_lines:
        word = line.split('/')[0]
        phs_seq = dict_words[word] #phonemes translation of word
        new_dev.write(f"{SC['src_dir']}/{line.strip()}\t{phs_seq}\n")
    new_dev.close()
    
    #Read lines from old test .txt
    old_test = open(SC['old_test_txt'], 'r')
    test_lines = old_test.readlines()
    old_test.close()
    random.shuffle(test_lines)
    
    #Create new test .txt (add src_dir and append phonemes-translation)
    new_test = open(SC['new_test_txt'], 'w')
    for line in test_lines:
        word = line.split('/')[0]
        phs_seq = dict_words[word] #phonemes translation of word
        new_test.write(f"{SC['src_dir']}/{line.strip()}\t{phs_seq}\n")
    new_test.close()
    
    #Get names of folders from {src_dir}
    src_folders = []
    for file in sorted(os.listdir(SC['src_dir'])):
        if '_' not in file and 'E' not in file:
            src_folders.append(file)
    
    #Create custom train .txt (with src_dir and phonemes-translation)
    train_lines = []
    general_counter = 0
    print("\tCreating Training .txt file...")
    for folder in src_folders:
        print(f"\t\tin folder {folder}", end='')
        folder_path = SC['src_dir'] + '/' + folder
        for audio in os.listdir(folder_path):
            general_counter += 1
            line = folder + '/' + audio + '\n'
            if line not in dev_lines and line not in test_lines:
                line = SC['src_dir'] + '/' + line.strip()
                phs_seq = dict_words[folder] #phoneme's translation
                train_lines.append(f"{line}\t{phs_seq}\n")
                
        print(" ...Finished")
    
    random.shuffle(train_lines)
    train_txt = open(SC['new_train_txt'], 'w')
    for line in train_lines:
        train_txt.write(line)
    train_txt.close()
    
    print("\t...Finished")

    print(f"Number of audios in Dev Dataset: {len(dev_lines)}")
    print(f"Number of audios in Test Dataset: {len(test_lines)}")
    print(f"Number of audios in Train Dataset: {len(train_lines)}")
    print(f"Total number of audios: {general_counter}")

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
    
    def keep_result(self):
        '''Determine whether or not to keep the given variables'''
        curr_ratio_loss = self.ratio_losses[-1]
        #If ratio loss is between 1.01 and 0.99 take it into consideration
        if curr_ratio_loss < 1.02 and curr_ratio_loss > 0.98:
            return True
        #Otherwise don't
        else:
            return False

class SC_DATASET(Dataset):
    """Generate dataset (tsv file) of two 'columns': path to spectrogram and
    IPA phoneme's translation"""
    def __init__(self, src, ipa2char, SC):
        #Initialize lists that will hold paths to spctrgrms and their texts
        self.spctrgrms_paths = []
        self.texts = []
        
        #Get the list of words that are in our phonemes-dictionary
        dict_words = pickle.load(open(SC['dict'], "rb" ))
        
        #Grab lines from src file (.txt)
        f = open(src, 'r')
        lines = f.readlines()
        f.close() 

        #Iterate through lines
        for line in lines:
            spctrgrm_path = SC['src_dir'] + '/' + line.strip()
            self.spctrgrms_paths.append(spctrgrm_path)
            
            #Grab word said in spectrogram; convert it to IPA phonemes
            word = spctrgrm_path.split('/')[-2]
            ipa_phones = dict_words[word]
            
            #Convert IPA phonemes to custom characters
            text = []        
            ipa_phones = ipa_phones.split(' ')
            for ph in ipa_phones:
                text.append(ipa2char[ph])

            self.texts.append('_'.join(text))

    def __getitem__(self, n):
        """Return path to spectrogram and IPA phoneme's translation"""
        spctrgrm_path = self.spctrgrms_paths[n]
        text = self.texts[n]
        return spctrgrm_path, text

    def __len__(self):
        return len(self.spctrgrms_paths)
      
def int_to_chars(ints_list, int2char):
    '''Convert list of ints to string of custom characters'''
    text = ''
    for i in ints_list:
        text += int2char[i]
    
    return text

def chars_to_int(text, char2int):
    '''Convert string of chars to list of integers'''
    integers = []
    text = text.replace(' ', '_ _')
    chars = text.split('_')
    for ch in chars:
        integers.append(char2int[ch])
                
    return integers

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
    
class BiGRU(nn.Module):
    def __init__(self, gru_dim, gru_hid_dim, gru_layers, dropout, batch_first):
        super(BiGRU, self).__init__()

        self.BiGRU = nn.GRU(input_size=gru_dim, hidden_size=gru_hid_dim,
            num_layers=gru_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, hparams):        
        super(SpeechRecognitionModel, self).__init__()
        #If we need to implement a second CNN, use this:
        #https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
        self.cnn1_kernel = hparams['cnn1_kernel']
        self.cnn1_stride = hparams['cnn1_stride']
        
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, hparams['cnn1_filters'], hparams['cnn1_kernel'], 
            stride=hparams['cnn1_stride'])
                               
        in_feats = hparams['cnn1_filters'] * (hparams['n_mels']//hparams['cnn1_stride'])
        self.fully_connected = nn.Linear(in_feats, hparams['gru_dim'])
        
        self.birnn_layers = BiGRU(hparams['gru_dim'], hparams['gru_hid_dim'],
            hparams['gru_layers'], hparams['gru_dropout'], batch_first=True)

        #Dynamiclly, set up value for classifier's out-ftrs of fc layers
        fc_out_ftrs = 0
        
        if hparams['gru_hid_dim']*2 <= hparams['n_class']:
            fc_out_ftrs = hparams['n_class']
        else:
            fc_out_ftrs = (hparams['gru_hid_dim']*2 - hparams['n_class']) // 2
            fc_out_ftrs += hparams['n_class']
        
        self.classifier = nn.Sequential(
            nn.Linear(hparams['gru_hid_dim']*2, fc_out_ftrs),  # birnn returns gru_hid_dim*2 (2 because bidirectional=True in nn.GRU)
            nn.GELU(),
            nn.Dropout(hparams['dropout']),
            nn.Linear(fc_out_ftrs, hparams['n_class'])
        )

    def add_paddings(self, x):
        #Pad if necessary
        paddings = [0,0,0,0] #left, right, top, bottom
        sizes = x.size()
        if (sizes[2] % 2) != 0: #H
            paddings[3] += 1
            
        if (sizes[3] % 2) != 0: #W
            paddings[1] += 1
        
        #Equation below won't work if cnn1_stride != 2
        pad_val = int(self.cnn1_kernel/self.cnn1_stride - 0.5)
        paddings = [val + pad_val for val in paddings]
        self.pad = nn.ZeroPad2d(paddings)

    def forward(self, x):
        x = self.pad(x)
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

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

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch,
          log_file, blank_label, int2char, char2ipa, losses):
    model.train()
    msg = f"\tEpoch: {epoch} | "
    
    batches_num = math.ceil(train_loader.batch_sampler.sampler.num_samples / 
                            train_loader.batch_size)
    #used as progress bar
    step = batches_num//10 + 1
    
    train_losses, lrs, train_per = [], [], []
    for idx, _data in enumerate(train_loader):
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
        
        if idx % step == 0:
            print(f"\tI have processed {idx+1}/{batches_num} batch(es)")
    
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
    
def save_chckpnt(best_model_wts, best_hparams, checkpoint_path, run_num,
                     epoch_num, optimizer_state_dict, train_log):
    '''Save models's weights and hyper parameters. Only if the model had a
    good ratio loss (between 0.99 and 1.01. Otherwise, don't save it.'''
    if run_num != '-1' and epoch_num != '-1':
        save_path = checkpoint_path[:-4] + f'_onRun{run_num.zfill(2)}'
        save_path += f'onEpoch{epoch_num.zfill(3)}' + checkpoint_path[-4:]
            
        #This is the best model of all epochs and all runs
        torch.save({
            'model_state_dict': best_model_wts,
            # 'optimizer_state_dict': optimizer_state_dict,
            'hparams': best_hparams
        }, save_path)
    else:
        msg = "\nMODEL DIDN'T HAVE A RATIO LOSS BETWEEN 1.02 AND 0.98. I AM "
        msg += "NOT SAVING THE MODEL.\n"
        log_message(msg, train_log, 'a', True)
        save_path = 'ModelNotSaved'
    
    return save_path
    
def log_labels(ipa2char, char2int, log_path):
    '''Log the conversions that took place from IPA to char and from char to
    int. In other words, log which integer (label) represented each phoneme'''
    msg = '\nThe label representation for all runs was the following:\n'
    msg += '   IPA char  int\n'
    for [k1, v1], [_, v2] in zip(ipa2char.items(), char2int.items()):
        msg += f"{k1:>5}{v1:>5}{v2:>5}\n"
    
    msg += f"'Blank Label': {len(ipa2char.items())}\n"
    log_message(msg, log_path, 'a', False)
    
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