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
import random
from random import randint
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import torchaudio
from torchaudio import transforms as tforms

def log_model_information(log_file, model, hparams):
    '''Log model summary, # of parameters, hyper parameters and more'''
    original_stdout = sys.stdout # Save a reference to the original std output
    with open(log_file, 'a') as log:
        sys.stdout = log # Change standard output to the created file
        #Number of parameters
        print('Total Number of Parameters in the Model is: ',end='')
        print(sum([param.nelement() for param in model.parameters()]))
        
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

def log_message(msg, log_file, mode, both=True):
    '''Function that prints and/or adds to log'''
    #Always log file
    with open(log_file, mode) as f:
        f.write(msg)
    
    #If {both} is true, print to terminal as well
    if both:
        print(msg, end='')

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

def plot_and_save(dev_losses, train_losses, cers, lrs, run, logs_folder):
    '''In one figure, plot and save dev_losses, train_losses and dev_cers; in
    a different figure, plot and save learning rate progress.'''
    fig_name = logs_folder + f'/zRun_{str(run).zfill(3)}.png'
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(f'Run {run}: Valid Loss, Train Loss and CER')
    x = list(range(1, len(dev_losses)+1))
    ax.plot(x, dev_losses, 'b', label="Validation Loss")
    ax.plot(x, train_losses, 'r', label="Train Loss")
    ax.plot(x, cers, 'g--', label="Char Error Rate")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    plt.savefig(fig_name)
    plt.show()
    
    #Learning Rate Progress
    fig_name = logs_folder + f'/zRun_{str(run).zfill(3)}_LR.png'
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title(f'Run {str(run).zfill(3)}: Learning Rate Progress')
    x = list(range(1, len(lrs)+1))
    ax.plot(x, lrs, 'g--', label="Learning Rate")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rates')
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
        self.lrs = []
        self.dev_losses = []
        self.cers = []
        self.ratio_losses = []
        
    def add_train_metrics(self, loss, lr):
        '''Add dev loss, ratio loss and cer'''   
        self.train_losses.append(loss)
        self.lrs.append(lr)
        
    def add_dev_metrics(self, loss, cer, ratio_loss):
        '''Add dev loss, ratio loss and cer'''   
        self.ratio_losses.append(ratio_loss)
        self.dev_losses.append(loss)
        self.cers.append(cer)
        
    def get_best_cer(self):
        '''Grab best CER'''
        return min(self.cers)
    
    def should_we_stop(self, epoch, early_stop):
        """If CER doesn't improve by p% in n epochs, stop training; where 
        n = early_stop['n'] and p = (1-early_stop['p'])*100"""
        stop, msg = False, ''
        
        if(epoch >= early_stop['n']):
            prev_cers = self.cers[-early_stop['n']:]
            if(prev_cers[0] * early_stop['p'] - min(prev_cers[1:]) < 0.00001):
                stop = True
                msg = 'EARLY STOP due to CER | '
            
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
    '''Convert list of ints to string of phonemes'''
    text = []
    for i in ints_list:
        text.append(int2char[i])
        
    text = '_'.join(text)
    text = text.replace('_ _', ' ')
    
    return text

def chars_to_ipa(predictions, targets, char2ipa):
    '''Trsanslate decoded predictions and decoded targets to ipa phonemes'''
    predicted_ipas, target_ipas = [], []
    
    for prediction in predictions:
        new_words = []
        for word in prediction.split(' '):
            new_word = []
            for ch in word.split('_'):
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
            for ch in word.split('_'):
                if ch == '':
                    new_word.append('')
                else:
                    new_word.append(char2ipa[ch])
                
            new_words.append('_'.join(new_word))
            
        target_ipas.append(' '.join(new_words))   
        
    return predicted_ipas, target_ipas


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


def data_processing(data, char2int, mels, FM=27, TM=0.125, CASE='dev'):
    spectrograms, labels, inp_lengths, label_lengths, filenames = [],[],[],[],[]
    
    for (spctrgrm_path, utterance) in data:   
        spec = torch.load(spctrgrm_path)
        #Apply audio transforms (frequency and time masking) to train samples
        if CASE == 'train':
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


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch,
          log_file, blank_label, int2char, char2ipa, losses):
    model.train()
    batches_num = train_loader.batch_sampler.num_of_batches
    # step = batches_num//2 + 1 #Used to print in terminal
    msg = f"\tEpoch: {epoch} | "
    
    avg_loss, decoded_preds, decoded_targets, MSG, lrs = 0, [], [], '', []
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths, filenames = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        
        model.add_paddings(spectrograms)
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        avg_loss += loss.detach().item()
        loss.backward()
        
        #To have an insight on the prediction of characters
        # decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1),
        #     labels, label_lengths, blank_label, int2char)
        
        #Decode prediction and targets into ipa phonemes (to assess progress)
        # predicted_ipas, target_ipas = chars_to_ipa(decoded_preds,
        #     decoded_targets, char2ipa)
       
        lrs.append(scheduler.get_lr()[0])        
        optimizer.step()
        scheduler.step()
        
        #Log some more predictions (first of these batches)
        # if batch_idx % step == 0:
        #     MSG += f'\t\tBatch [{batch_idx+1}/{batches_num}] | '
        #     MSG += f"{target_ipas[0]}' -> '{predicted_ipas[0]}'\n"
            
    avg_loss /= batches_num
    avg_lr = sum(lrs) / batches_num
    losses.add_train_metrics(avg_loss, avg_lr)
    
    msg += f"Train Avg Loss: {avg_loss:.4f}\n"
    # msg += f"Train Avg Loss: {avg_loss:.4f} | '{target_ipas[0]}' -> "
    # msg += f"'{predicted_ipas[0]}'\n"
    log_message(msg, log_file, 'a', True)
    # log_message(MSG, log_file, 'a', False)
            
def dev(model, device, dev_loader, criterion, epoch, log_file, blank_label,
        int2char, char2ipa, metrics):
    
    batches_num = dev_loader.batch_sampler.num_of_batches
    #used to print decoded predictions and targets
    step = batches_num//2 + 1
    
    model.eval()
    dev_loss = 0
    dev_cer, dev_wer = [], []
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
            dev_loss += loss.detach().item()

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
                dev_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                dev_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(dev_cer)/len(dev_cer)
    avg_wer = sum(dev_wer)/len(dev_wer)
    dev_loss /= batches_num
    ratio_loss = dev_loss / metrics.train_losses[epoch-1]
    msg += f"Dev Avg Loss: {dev_loss:.4f} | Ratio Loss: {ratio_loss:.4f} | "
    msg += f"Avg CER: {avg_cer:.4f} | Avg WER: {avg_wer:.4f}\n"
    
    log_message(msg, log_file, 'a', True)
    log_message(MSG, log_file, 'a', True)
    metrics.add_dev_metrics(dev_loss, avg_cer, ratio_loss)

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
    
def find_best_lr(model, loss_fn, optimizer, train_loader, init_value,
                 final_value, device):
    '''Implement 'Programming PyTorch for Deep Learning' method of finding
    best learning rate. Plot graph for visual aid.'''
    lrs_to_plot = []
    losses_to_plot = []
    
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, targets, input_lengths, label_lengths, filenames = data 
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        model.add_paddings(inputs)
        outputs = model(inputs)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1) # (time, batch, n_class)
        loss = loss_fn(outputs, targets, input_lengths, label_lengths)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            if(len(log_lrs) > 20):
                lrs_to_plot = log_lrs[10:-2]
                losses_to_plot = losses[10:-2]
                break
            
            else:
                lrs_to_plot = log_lrs
                losses_to_plot = losses
                break

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss.item())
        #log_lrs.append((lr)) #TYPO!!!
        log_lrs.append(math.log10(lr))

        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
        
    if(len(log_lrs) > 20):
        lrs_to_plot = log_lrs[10:-2]
        losses_to_plot = losses[10:-2]
        
    else:
        lrs_to_plot = log_lrs
        losses_to_plot = losses
        
    #Plot learning rates (log scale) vs losses
    fig, ax = plt.subplots()  # a figure with a single Axes
    ax.set_title('Learning Rates vs. Losses')
    ax.plot(lrs_to_plot, losses_to_plot, 'b')
    ax.grid(True)
    ax.set_xlabel('Learning Rates (Log Scale)')
    ax.set_ylabel('Losses')
    plt.show()

class BucketsSampler(Sampler):
    '''Batch sampler that groups audios by their duration. It shuffles the
    audios inside each bucket and shuffles the batches once all audios have
    been assigned a bucket and a batch. {bucket_boundaries} determines how
    many buckets you will have and which boundaries they will have.'''
    def __init__(self, transcript, bucket_boundaries, batch_size, drop_last):
        #Create a Dictionary of upper and lower boundaries with respective ID
        lower_bounds = [0] + bucket_boundaries
        upper_bounds = bucket_boundaries + [math.inf]
        self.Buckets = {}
        for ID, _ in enumerate(lower_bounds):
            self.Buckets[ID] = [lower_bounds[ID], upper_bounds[ID]]
        
        #Determine which bucket each audio pertains to using duration of audio
        f = open(transcript, 'r')
        data_buckets = dict()
        for audio_idx, line in enumerate(f):
            _, _, duration = line.strip().split('\t')
            bucket_id = self.get_bucket_id(audio_idx, int(duration))
            if bucket_id in data_buckets.keys():
                data_buckets[bucket_id].append(audio_idx)
            else:
                data_buckets[bucket_id] = [audio_idx]
        
        f.close()
        
        #Create batches for each bucket
        self.drop_last = drop_last
        self.bs = batch_size
        self.data_buckets = data_buckets
        self.create_batches()
                
    def __iter__(self):
        self.create_batches() #Run again to re-random audios and batches
        random.shuffle(self.batches)
        for batch in self.batches: 
            yield batch
    
    def __len__(self):
        '''I am returning the number of batches in the Sampler, not the number
        of samples in the dataset.'''
        return self.num_of_batches
    
    def create_batches(self):
        #Create batches from buckets
        batches = []
        for bucket_id, audio_indices in self.data_buckets.items():
            #Shuffle the audios in bucket
            random.shuffle(audio_indices)
            #Break audio_indices into sublists of size {bs}; append them to batches
            batches += [audio_indices[i * self.bs:(i + 1) * self.bs] for i in
                        range((len(audio_indices) + self.bs - 1) // self.bs)]  
            
            if self.drop_last:
                if len(batches[-1]) != self.bs:
                    batches.pop()
                    
        self.num_of_batches = len(batches)
        self.batches = batches
    
    def get_bucket_id(self, index, duration):
        '''Determine which bucket the audio pertains to using duration'''
        for ID, [lower_bound, upper_bound] in self.Buckets.items():
            if lower_bound <= duration <= upper_bound:
                return ID
                
        print(f"ERROR: Audio with index {index} has a duration of "
              "{float(duration/100)} seconds. Not in the buckets' boundaries.")
        sys.exit()
