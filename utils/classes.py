'''/****************************************************************************
    File: utis/classes.py
    Author(s): Mario Esparza
    Date: 03/11/2022
    Description: Includes a variety of classes like Manager (to manage a tester
        or a trainer); Custom_Dataset; Metrics (to keep track error rates and
        losses); Lr_Sched (to manage custom learning rate scheduler); and Logger
    Updates: N/A
        
*****************************************************************************'''
import json
import matplotlib.pyplot as plt
import os.path as os_path
import pickle
import string
import sys
import torch

from torch.utils.data import Dataset

class Manager():
    '''To manage a trainer or tester (this is their parent class).'''
    def __init__(self):
        self.reserved_chars = [' ']

        self.ipa2char = {}
        self.char2ipa = {}
        self.int2char = {}
        self.char2int = {}

        self.HP = {} #Hyper parameters
        self.OP = {} #Other parameters

        self.use_cuda = False
        self.device = torch.device("cpu")
        self.data_loader_kwargs = {}

    def get_custom_chars(self):
        '''Get list of custom characters that will replace IPA phonemes'''
        chars = []
        for c in string.ascii_lowercase:
            chars.append(c)
            
        for c in string.ascii_uppercase:
            chars.append(c)
            
        for c in range(0,10):
            chars.append(str(c))
        
        return chars

    def get_ipa_phonemes_list(self, path_to_phonemes):
        '''Get unique list of phonemes that exist in given file'''
        IPA_phones = []
        words_in_dict = pickle.load(open(path_to_phonemes, "rb" ))
        for phonemes in list(words_in_dict.values()):
            for ph in phonemes.split('_'):
                if ph not in IPA_phones:
                    IPA_phones.append(ph)
                
        return sorted(IPA_phones)

    def set_mappings_dictionaries(self, path_to_phonemes):
        '''Returns IPA to Char, Char to IPA, Char to Int, and Int to Char 
        dictionaries'''
        # Get a unique list of IPA phonemes.
        IPA_phones = self.get_ipa_phonemes_list(path_to_phonemes)
        #Get list of characters that will replace IPA phonemes in csvs
        custom_chars = self.get_custom_chars()
        
        #Check that I have enough custom characters
        if len(IPA_phones) > len(custom_chars):
            print("You don't have enough custom characters :( ")
            sys.exit()

        #Add reserved characters at the beginning of both lists
        for idx, ch in enumerate(self.reserved_chars):
            IPA_phones.insert(idx, ch)
            custom_chars.insert(idx, ch)
            
        #Create ipa-to-character and character-to-ipa maps (dictionaries)
        for ph, ch in zip(IPA_phones, custom_chars):
            self.ipa2char[ph] = ch
            self.char2ipa[ch] = ph
            
        #Create int-to-char and char-to-int maps (dictionaries)
        for idx, ch in enumerate(list(self.char2ipa.keys())):
            self.int2char[idx] = ch
            self.char2int[ch] = idx

    def fix_ints_lists(self, raw_hp):
        '''Convert strings that came from json to lists of integers'''
        ints = raw_hp["ints"]
        for k,v in ints.items():
            self.HP[k] = [int(x) for x in v.split(',') if x != ""]

    def fix_floats_lists(self, raw_hp):
        '''Convert strings that came from json to lists of floats'''
        ints = raw_hp["floats"]
        for k,v in ints.items():
            self.HP[k] = [float(x) for x in v.split(',') if x != ""]

    def fix_ints(self, raw_op):
        '''Convert strings that came from json to integers'''
        ints = raw_op["ints"]
        for k,v in ints.items():
            self.OP[k] = int(v)

    def fix_floats(self, raw_op):
        '''Convert strings that came from json to floats'''
        floats = raw_op["floats"]
        for k,v in floats.items():
            self.OP[k] = float(v)

    def fix_booleans(self, raw_op):
        '''Convert strings that came from json to booleans'''
        booleans = raw_op["booleans"]
        for k,v in booleans.items():
            self.OP[k] = True if v == "True" else False

    def set_hyperparams(self, path_to_json):
        '''Convert hyperparameters from JSON to Python's datatypes'''
        with open(path_to_json, "r") as json_file:
            raw_hp = json.load(json_file)
            self.fix_ints_lists(raw_hp)
            self.fix_floats_lists(raw_hp)

    def set_otherparams(self, path_to_json):
        '''Convert 'other' parameters from JSON to Python's datatypes'''
        with open(path_to_json, "r") as json_file:
            raw_op = json.load(json_file)
            self.fix_ints(raw_op)
            self.fix_floats(raw_op)
            self.fix_booleans(raw_op)

        self.OP["blank_label"] = len(self.ipa2char.keys()) #for CTC
        self.OP["n_class"] = self.OP["blank_label"] + 1 #number of labels

    def set_cuda_and_device(self):
        '''Use GPU if available'''
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.data_loader_kwargs = {'num_workers': 1, 'pin_memory': True}

class Custom_Dataset(Dataset):
    """Dataset that contains spectrograms' paths and texts"""
    def __init__(self, csv_file, ipa2char, ratio, direction):
        #Initialize lists that will hold texts and paths to spectrograms 
        self.spctrgrms_paths = []
        self.texts = []
        
        #Get lines frrom csv_file
        f = open(csv_file, 'r')
        lines = f.readlines()

        #Determine number of samples to grab and where to start grabbing from
        if(direction == "front"):
            lines = lines[:int(len(lines) * ratio)]
        elif(direction == "back"):
            lines = lines[int(len(lines) * ratio):]
        else:
            print(f"What do you mean by direction = '{direction}'")
            sys.exit()

        #Iterate through csv file and get spctrgrms_paths and texts
        for line in lines:
            spctrgrm_path, text = line.strip().split(',')
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
                        
        f.close() 

    def __getitem__(self, n):
        """Return spectrogram path and text of the nth item"""
        spctrgrm_path = self.spctrgrms_paths[n]
        text = self.texts[n]
        return spctrgrm_path, text

    def __len__(self):
        return len(self.spctrgrms_paths)

class Metrics:
    """To keep metrics across training and validation"""
    def __init__(self):
        self.train_losses = []
        self.train_pers = []
        self.lrs = []
        self.dev_losses = []
        self.dev_pers = [] #Phoneme error rates
        self.ratio_losses = []

        #Variables related to best PER given a great ratio loss
        self.RL_global_best_per = 2.0
        self.RL_best_model_wts = {}
        self.RL_best_hparams = {}
        self.RL_run_num = '-1'
        self.RL_epoch_num = '-1'
    
        #Variables related to best PER of all (no caring about anything else)
        self.best_pers = []
        self.global_best_per = 2.0
        self.best_model_wts = {}
        self.best_hparams = {}
        self.run_num = '-1'
        self.epoch_num = '-1'
        
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
    
    def should_we_stop(self, epoch, vals):
        """If PER doesn't improve by %p in n epochs, stop training; where 
        n = early_stop_n and p = (1-early_stop_p)*100. On the other
        hand, if we have overfitting above t for n epochs, stop training.
        Where t = early_stop_t. Start checking for early stop once w epochs
        have passed; where w = early_stop_w"""
        stop, msg = False, ''
        
        if(epoch >= vals["early_stop_w"]):
            prev_pers = self.dev_pers[-vals["early_stop_n"]:]
            if(prev_pers[0] * vals["early_stop_p"] - min(prev_pers[1:]) < 0.00001):
                stop = True
                msg = 'EARLY STOP due to PER | '
            
            #TEMPORARILY, I will do early stop using PER only
            # #n previous ratio losses
            # ratio_losses = self.ratio_losses[-vals["early_stop_n"]:]
            # #If all of them are above threshold t, stop due to overfitting
            # counter = [1 if i > vals["early_stop_t"] else 0 for i in ratio_losses]
            # if sum(counter) == vals["early_stop_n"]:
            #     stop = True
            #     msg = 'EARLY STOP due to OVERFIT | '
                
        return stop, msg
    
    def keep_RL_result(self):
        '''Determine whether or not to keep checkpoint given the ratio loss'''
        curr_ratio_loss = self.ratio_losses[-1]
        #If ratio loss is between 1.01 and 0.99 take it into consideration
        if curr_ratio_loss < 1.02 and curr_ratio_loss > 0.98:
            return True
        #Otherwise don't
        else:
            return False

class Lr_Sched():
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

class Logger():
    def __init__(self, save_dir, file_name, initial_msg):
        #Set path where logs will be saved
        self.parent_dir = save_dir
        self.log_path = os_path.join(save_dir, file_name)
        self.create_file(initial_msg)

        self.RED = '\033[31m' #Used to print red messages in terminal
        self.PURPLE = '\033[35m' #Used to print purple messages in terminal
        self.WHITE = '\033[0m'  #Used to print white messages in terminal
        self.GREEN = '\033[32m' #Used to print green messages in terminal

    def add_color(self, msg, color):
        if color == "RED":
            msg =  self.RED + msg + self.WHITE
        
        return msg

    def create_file(self, msg):
        #Create file, overwrite if already exists
        msg = f"{msg} '{self.log_path}'\n"
        self.log_msg(msg, 'w', True) 

    def log_msg(self, msg: str, mode: str, both: bool, color = "") -> None:
        '''Function that prints and/or adds to log'''
        #Always log into file
        with open(self.log_path, mode) as f:
            f.write(msg)
        
        #If {both} is true, print to terminal too
        if both:
            if color:
                msg = self.add_color(msg, color)

            print(msg, end='')

    def save_losses_fig(self, file_name, metrics, param_grid_idx):
        '''Save plot of validation and training losses'''
        fig_name = os_path.join(self.parent_dir, file_name + '_DevVsTrain.png')
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()  # a figure with a single Axes
        ax.set_title(f'Run {param_grid_idx}: Valid Loss vs. Train Loss')
        x = list(range(1, len(metrics.dev_losses)+1))
        ax.plot(x, metrics.dev_losses, 'b', label="Validation Loss")
        ax.plot(x, metrics.train_losses, 'r', label="Train Loss")
        ax.grid(True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Losses')
        ax.legend(loc='upper center', shadow=True, fontsize='small')
        plt.tight_layout() #gives space to x label in the .png file
        plt.savefig(fig_name)
        # plt.show()

    def save_pers_fig(self, file_name, metrics, param_grid_idx):
        '''Plot and save validation phoneme error rates'''
        fig_name = os_path.join(self.parent_dir, file_name + '_PER.png')
        fig, ax = plt.subplots()  # a figure with a single Axes
        ax.set_title(f'Run {param_grid_idx}: Phoneme Error Rates (PERs)')
        x = list(range(1, len(metrics.dev_pers)+1))
        ax.plot(x, metrics.dev_pers, 'b', label="Validation PERs")
        ax.plot(x, metrics.train_pers, 'r', label="Train PERs")
        ax.grid(True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('PERs')
        ax.legend(loc='upper center', shadow=True, fontsize='small')
        plt.tight_layout()
        plt.savefig(fig_name)
        # plt.show()

    def save_learning_rates_fig(self, file_name, metrics, param_grid_idx):
        '''Plot and save learning rates process'''
        fig_name = os_path.join(self.parent_dir, file_name + '_LR.png')
        fig, ax = plt.subplots()  # a figure with a single Axes
        ax.set_title(f'Run {param_grid_idx}: Learning Rate Progress')
        x = list(range(1, len(metrics.lrs)+1))
        ax.plot(x, metrics.lrs, 'g--', label="Learning Rate")
        ax.grid(True)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rates')
        ax.legend(loc='upper center', shadow=True, fontsize='small')
        plt.tight_layout()
        plt.savefig(fig_name)
        # plt.show()

    def save_figs(self, metrics, param_grid_idx, ):
        '''In figure one, plot and save dev_losses vs train_losses. In figure two,
        plot and save validation PERs vs train PERs. In figure three, plot and
        save progress of learning rate.'''
        file_name = f'zRun_{str(param_grid_idx).zfill(3)}'
        
        self.save_losses_fig(file_name, metrics, param_grid_idx)
        self.save_pers_fig(file_name, metrics, param_grid_idx)
        self.save_learning_rates_fig(file_name, metrics, param_grid_idx)

        m = "Plots of losses, phoneme error rates and learning rates have "
        m += f"been saved here: {self.parent_dir}.\n"
        self.log_msg(m, 'a', True)
        