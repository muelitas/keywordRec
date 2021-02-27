'''/**************************************************************************
    File: preprocessing.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/26/2021
    
    TODO
    
***************************************************************************''' 
import os
import pickle
import random
import string
import sys
import torchaudio

from utils import delete_contents

def check_folder(this_path):
    '''If {this_path} exists, ask if okay to overwrite; otherwise create it'''
    folder = ''
    subfolder = this_path
    #Check if {this_path} is either a folder or a folder/subfolder situation
    if '/' in subfolder:
        folder = '/'.join(this_path.split('/')[:-1])
    
    #If this is a folder/subfolder situation 
    if len(folder) != 0:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    #If subfolder doesn't exist, create it
    if not os.path.isdir(subfolder):
            os.mkdir(subfolder)    
    
    if len(os.listdir(subfolder)) != 0:
        print(f"{subfolder} isn't empty, is it okay if I overwrite it?")
        if input() != 'y':
            sys.exit()
        else:
            delete_contents(subfolder)

def k_words_stats(dev_csv, k_words, phones_dict):
    """Iterate through each line in {dev_csv}. Determine how many instances of
    k_words were found in {dev_csv}"""
    #Get dictionary of words and their phoneme representation
    dict_words = pickle.load(open(phones_dict, "rb" ))  
    
    #Convert k_words to sets of phonemes splitted by underscores
    K_words = []
    for k_word in k_words:
        K_words.append(dict_words[k_word].replace(' ', '_'))
    
    #Keeps track of how many instances of k_words were found in {dev_csv}
    GT_num = 0
    with open(dev_csv, 'r') as F:
        for ind, line in enumerate(F):
            words = line.strip().split('\t')[1].split(' ')
            for idx, word in enumerate(words):
                if word in K_words:
                    GT_num += 1
                    
    print(f"\nThe number of k_words found in dev dataset is {GT_num}\n")
    return GT_num

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
      
def merge_csvs(datasets, final_csv, Case):
    """Iterate through {datasets}. If indicated to use, depending on {Case},
    grab respective csv file and append its lines in {final_csv}."""
    total_time = 0.0
    counter = 0
    
    F = open(final_csv, 'w')
    for _set in datasets:
        #If indicated to use dataset, get train or dev csv (depends on {Case})
        if _set['use_dataset']:
            csv = _set['train_csv'] if Case == 'train' else _set['dev_csv']
            #Scan csv, get audios' duration, add line to {final_csv}
            f = open(csv, 'r')
            for idx, line in enumerate(f):   
                _, _, duration = line.strip().split('\t')
                total_time += float(duration)/1000 #(/1000) to convert to secs
                F.write(line)
                
            f.close()
            counter += (idx + 1)
            print(f"\n'{csv.split('/')[-1]}' has been added to "
                  f"'{final_csv.split('/')[-1]}'")
        
    F.close()
    
    print(f"\nAfter merging, '{final_csv.split('/')[-1]}' is composed of "
          f"{counter} lines ({total_time:.2f} seconds).")
    
def get_custom_chars(manual_chars):
    '''Get list of custom characters that will replace IPA phonemes'''
    chars = []
    for c in string.ascii_lowercase:
        chars.append(c)
        
    for c in string.ascii_uppercase:
        chars.append(c)
        
    for c in range(0,10):
        chars.append(str(c))
        
    for c in manual_chars:
        chars.append(str(c))
    
    return chars

def get_mappings(datasets, other_chars, manual_chars):
    '''Returns IPA to Char, Char to IPA, Char to Int and Int to Char 
    dictionaries'''
    # Get a unique list of IPA phonemes present in our dictionaries. Only from
    # those dictionaries that we wish to use
    IPA_phones = []
    for dataset in datasets:
        if dataset['use_dataset']:
            words_in_dict = pickle.load(open(dataset['dict'], "rb" ))
            for phonemes in list(words_in_dict.values()):
                for ph in phonemes.split(' '):
                    if ph not in IPA_phones:
                        IPA_phones.append(ph)
                
    IPA_phones = sorted(IPA_phones)
    
    #Get list of characters that will replace IPA phonemes in csvs
    custom_chars = get_custom_chars(manual_chars)
    
    #Check that I have enough custom characters
    if len(IPA_phones) > len(custom_chars):
        ask_user("You don't have enough custom characters :( ")
        
    #Add special (other) characters at the beginning of both lists
    for idx, ch in enumerate(other_chars):
        IPA_phones.insert(idx, ch)
        custom_chars.insert(idx, ch)
        
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

def TS_create_csv(dataset):
    '''Iterates through lines in {transcript} and creates 2 transcripts
    (train & dev) that contain three columns: pt_full_path, audio's text and
    audio's duration. It also translates each word in the transcripts into a
    set of IPA phonemes'''
    #Get dictionary of words and their phoneme representation
    dict_words = pickle.load(open(dataset['dict'], "rb" ))
    
    #Grab all lines from transcript
    f = open(dataset['transcript'], 'r')
    lines = f.readlines()
    f.close()
    
    #If desired number of audios to use exceeds available audios, give warning
    if((dataset['num'] != None) and (dataset['num'] >= len(lines))):
        print(f"\nWARNING: {dataset['num']} is out of range for TS's dataset."
              " I am setting 'num' equal to 'None'. This will give you all "
              "the audios in the Dataset.")
        dataset['num'] = None
    
    #Determine how many samples I will need for validation
    num = len(lines) if dataset['num'] == None else dataset['num']
    dev_split = int(dataset['splits'][1] * num)
    
    #No need to shuffle here. I shuffled back when I merged TTS and gTTS and
    #am shuffling in Batch Sampler.
    
    #Get training samples; save tensor paths and their phonemes in {train_csv}
    f = open(dataset['train_csv'], 'w')
    for item in lines[dev_split:dataset['num']]:
        wav_path, word, duration = item.split('\t')
        phs_seq = dict_words[word] #phonemes translation of word
        phs_seq = phs_seq.replace(' ', '_')
        f.write(wav_path + '\t' + phs_seq + '\t' + duration)
    f.close()
    
    #Get validation samples; save tensor paths and their phonemes in {dev_csv}
    f = open(dataset['dev_csv'], 'w')
    for item in lines[:dev_split]:
        wav_path, word, duration = item.split('\t')
        phs_seq = dict_words[word] #phonemes translation of word
        phs_seq = phs_seq.replace(' ', '_')
        f.write(wav_path + '\t' + phs_seq + '\t' + duration)
    f.close()
    
    print(f"\nThe files '{dataset['train_csv'].split('/')[-1]}' and "
          f"'{dataset['dev_csv'].split('/')[-1]}' have been created; "
          f"{len(lines[dev_split:dataset['num']])} and {dev_split}"
          " lines have been added to each file respectively.")

def KA_create_csv(dataset):
    '''Iterates through lines in {transcript} and creates 2 transcripts
    (train & dev) that contain three columns: pt_full_path, audio's text and
    audio's duration. It also translates each word in the transcripts into a
    set of IPA phonemes'''
    #Get dictionary of words and their phoneme representation
    dict_words = pickle.load(open(dataset['dict'], "rb" ))
    #Grab all lines from transcript
    f = open(dataset['transcript'], 'r')
    lines = f.readlines()
    f.close()
    #Determine how many samples I will need for validation
    dev_split = int(dataset['splits'][1] * len(lines))
    
    #BTW, no need to shuffle. I am shuffling in dataloaders
    
    #Get training samples; save tensor paths and their phonemes in {train_csv}
    f = open(dataset['train_csv'], 'w')
    for item in lines[dev_split:]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    #Get validation samples; save tensor paths and their phonemes in {dev_csv}
    f = open(dataset['dev_csv'], 'w')
    for item in lines[:dev_split]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    print(f"\nThe files '{dataset['train_csv'].split('/')[-1]}' and "
          f"'{dataset['dev_csv'].split('/')[-1]}' have been created; "
          f"{len(lines[dev_split:])} and {dev_split}"
          " lines have been added to each file respectively.")
    
def TI_create_csv(TI_data, phones_dict):
    '''Iterates through lines in {transcript} and creates 2 transcripts
    (train & dev) that contain three columns: pt_full_path, audio's text and
    audio's duration. It also translates each word in the transcripts into a
    set of IPA phonemes'''
    #Get dictionary of words and their phoneme representation
    dict_words = pickle.load(open(phones_dict, "rb" ))
    #Grab all lines from transcript
    f = open(TI_data['transcript'], 'r')
    lines = f.readlines()
    f.close()
    #Determine how many samples I will need for validation
    dev_split = int(TI_data['splits'][1] * len(lines))
    
    #BTW, no need to shuffle. I am shuffling in dataloaders
    
    #Get training samples; save tensor paths and their phonemes in {train_csv}
    f = open(TI_data['train_csv'], 'w')
    for item in lines[dev_split:]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    #Get validation samples; save tensor paths and their phonemes in {dev_csv}
    f = open(TI_data['dev_csv'], 'w')
    for item in lines[:dev_split]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    print(f"\nThe files '{TI_data['train_csv'].split('/')[-1]}' and "
          f"'{TI_data['dev_csv'].split('/')[-1]}' have been created; "
          f"{len(lines[dev_split:])} and {dev_split}"
          " lines have been added to each file respectively.")
    
def LS_create_csv(LS_data, phones_dict):
    '''Iterates through lines in {transcript} and creates 2 transcripts
    (train & dev) that contain three columns: pt_full_path, audio's text and
    audio's duration. It also translates each word in the transcripts into a
    set of IPA phonemes'''
    #Get dictionary of words and their phoneme representation
    dict_words = pickle.load(open(phones_dict, "rb" ))
    
    #Grab all lines from transcript
    f = open(LS_data['transcript'], 'r')
    lines = f.readlines()
    f.close()
    
    #If desired number of audios-to-use exceeds available audios, give warning
    if((LS_data['num'] != None) and (LS_data['num'] >= len(lines))):
        print(f"\nWARNING: {LS_data['num']} is out of range for TS's dataset."
              " I am setting 'num' equal to 'None'. This will give you all "
              "the audios in the Dataset.")
        LS_data['num'] = None
    
    #Determine how many samples I will need for validation
    num = len(lines) if LS_data['num'] == None else LS_data['num']
    dev_split = int(LS_data['splits'][1] * num)
    
    #BTW, no need to shuffle. I got audios randomly and I am shuffling in
    #dataloaders
    
    #Get training samples; save tensor paths and their phonemes in {train_csv}
    f = open(LS_data['train_csv'], 'w')
    for item in lines[dev_split:LS_data['num']]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    #Get validation samples; save tensor paths and their phonemes in {dev_csv}
    f = open(LS_data['dev_csv'], 'w')
    for item in lines[:dev_split]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    print(f"\nThe files '{LS_data['train_csv'].split('/')[-1]}' and "
          f"'{LS_data['dev_csv'].split('/')[-1]}' have been created; "
          f"{len(lines[dev_split:LS_data['num']])} and {dev_split}"
          " lines have been added to each file respectively.")

def TS_check(k_words, dataset):
    '''Keep track of how many times each keyword appears in {transcript}.
    Also, determine if all words in {transcript} exist in our dictionary
    {phones_dict}. If some word(s) don't exist, flag and notify.'''
    #To keep track of the number of instances for each key word
    kword_dict = {}
    for k_word in k_words:
        kword_dict[k_word] = 0
        
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(dataset['dict'], "rb" ))
    words_in_dict = list(dict_words.keys())
    words_not_in_dict, found_here =  [], []
    del dict_words
    
    with open(dataset['transcript'], 'r') as transcr:
        for line in transcr:
            word = line.split('\t')[1]
            #If word is in k_words, add number of instances
            if word in k_words:
                kword_dict[word] += 1
                
            #If word not in dictionary, save it so we can add it later
            if word not in words_in_dict and word not in words_not_in_dict:
                words_not_in_dict.append(word)
                found_here.append(line)
                
    #Display number of instances found for each keyword
    print(f"\nIn '{dataset['transcript']}', I found:")
    for k,v in kword_dict.items():
        print(f"\t{v} instances of the word '{k}'")
    
    if len(words_not_in_dict) != 0:
        print(f"\tI also found {len(words_not_in_dict)} words that are not in"
               " our phonemes-dictionary. Such words are:")
        
        for idx, w in enumerate(words_not_in_dict):
            print(f"\t\t- {w} found here {found_here[idx]}")
            
    else:
        print("\nAll words are in our dataset's dictionary. No manual "
              "additions needed for this dataset.")

def KA_check(k_words, dataset):
    '''Keep track of how many times each keyword appears in {transcript}.
    Also, determine if all words in {transcript} exist in our dictionary
    {phones_dict}. If some word(s) don't exist, flag and notify.'''
    #To keep track of the number of instances for each key word
    kword_dict = {}
    for k_word in k_words:
        kword_dict[k_word] = 0
        
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(dataset['dict'], "rb" ))
    words_in_dict = list(dict_words.keys())
    words_not_in_dict, found_here =  [], []
    del dict_words
    
    with open(dataset['transcript'], 'r') as transcr:
        for line in transcr:
            sentence = line.split('\t')[1]
            words = sentence.split(' ')
            for word in words:
                #If word is in k_words, add number of instances
                if word in k_words:
                    kword_dict[word] += 1
            
                #If word not in dictionary, save it so we can add it later
                if word not in words_in_dict and word not in words_not_in_dict:
                    words_not_in_dict.append(word)
                    found_here.append(line)
                
    #Display number of instances found for each keyword
    print(f"\nIn '{dataset['transcript']}', I found:")
    for k,v in kword_dict.items():
        print(f"\t{v} instances of the word '{k}'")
    
    if len(words_not_in_dict) != 0:
        print(f"\tI also found {len(words_not_in_dict)} words that are not in"
               " our phonemes-dictionary. Such words are:")
        
        for idx, w in enumerate(words_not_in_dict):
            print(f"\t\t- {w} found here {found_here[idx]}")
    else:
        print("\nAll words are in our dataset's dictionary. No manual additions"
              " needed for this dataset.")
 
def TI_check(k_words, transcript, phones_dict):
    '''Keep track of how many times each keyword appears in {transcript}.
    Also, determine if all words in {transcript} exist in our dictionary
    {phones_dict}. If some word(s) don't exist, flag and notify.'''
    #To keep track of the number of instances for each key word
    kword_dict = {}
    for k_word in k_words:
        kword_dict[k_word] = 0
        
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(phones_dict, "rb" ))
    words_in_dict = list(dict_words.keys())
    words_not_in_dict, found_here =  [], []
    del dict_words
    
    with open(transcript, 'r') as f:
        for line in f:
            sentence = line.split('\t')[1]
            words = sentence.split(' ')
            for word in words:
                #If word is in k_words, add number of instances
                if word in k_words:
                    kword_dict[word] += 1
            
                #If word not in dictionary, save it so we can add it later
                if word not in words_in_dict and word not in words_not_in_dict:
                    words_not_in_dict.append(word)
                    found_here.append(line)
                
    #Display number of instances found for each keyword
    print(f"\nIn '{transcript}', I found:")
    for k,v in kword_dict.items():
        print(f"\t{v} instances of the word '{k}'")
    
    if len(words_not_in_dict) != 0:
        print(f"\tI also found {len(words_not_in_dict)} words that are not in"
               " our phonemes-dictionary. Such words are:")
        print("\nTo add these words to your dict., use run_09/spanish_dict")
        
        for idx, w in enumerate(words_not_in_dict):
            print(f"\t\t- {w} found here {found_here[idx]}")
            
        print("\nHere are the words in a 'list' format, this way you can copy "
              f"and paste: {words_not_in_dict}")
    else:
        print("\nAll words are in our phonemes-dictionary. No manual additions "
              "needed for this dataset.")

def LS_check(k_words, transcript, phones_dict):
    '''Keep track of how many times each keyword appears in {transcript}.
    Also, determine if all words in {transcript} exist in our dictionary
    {phones_dict}. If some word(s) don't exist, flag and notify.'''
    #To keep track of the number of instances for each key word
    kword_dict = {}
    for k_word in k_words:
        kword_dict[k_word] = 0
        
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(phones_dict, "rb" ))
    words_in_dict = list(dict_words.keys())
    words_not_in_dict, found_here =  [], []
    del dict_words
    
    with open(transcript, 'r') as f:
        for line in f:
            sentence = line.split('\t')[1]
            words = sentence.split(' ')
            for word in words:
                #If word is in k_words, add number of instances
                if word in k_words:
                    kword_dict[word] += 1
            
                #If word not in dictionary, save it so we can add it later
                if word not in words_in_dict and word not in words_not_in_dict:
                    words_not_in_dict.append(word)
                    found_here.append(line)
                
    #Display number of instances found for each keyword
    print(f"\nIn '{transcript}', I found:")
    for k,v in kword_dict.items():
        print(f"\t{v} instances of the word '{k}'")
    
    if len(words_not_in_dict) != 0:
        print(f"\tI also found {len(words_not_in_dict)} words that are not in"
               " our phonemes-dictionary. Such words are:")
        print("\nTo add these words to your dict., use run_09/spanish_dict")
        
        for idx, w in enumerate(words_not_in_dict):
            print(f"\t\t- {w} found here {found_here[idx]}")
            
        print("\nHere are the words in a 'list' format, this way you can copy "
              f"and paste: {words_not_in_dict}")
    else:
        print("\nAll words are in our phonemes-dictionary. No manual additions "
              "needed for this dataset.")
        
def SC_check_and_create_csv(k_words, SC, phones_dict):
    '''Keep track of how many times each keyword appears in {src_dir}.
    Determine if all words in {src_dir} exist in {phones_dict}. If some word(s)
    don't exist, flag and notify. Iterate through files in {src_dir} and create
    2 transcripts (train & dev) that contain three columns: pt_full_path,
    audio's text and audio's duration. Translate each word into a set of IPA
    phonemes'''
    print("Processing Speech Commands' Dataset...")
    
    if SC['num'] > 54495 or SC['num'] == None:
        SC['num'] = 54495
        print("\tNOTE: 'num' is too high. I am setting num = 54495 to keep "
              "dataset balanced")
        
    if SC['num'] % 35 != 0:
        SC['num'] -= SC['num'] % 35
        print("\tNOTE: 'num' is not a multiple of 35. I am setting it equal "
              f"to {SC['num']}, to keep dataset balanced")
        
    #To keep track of the number of instances for each key word
    kword_dict = {}
    for k_word in k_words:
        kword_dict[k_word] = 0
    
    #Get the list of words that are in our phonemes-dictionary
    dict_words = pickle.load(open(phones_dict, "rb" ))
    words_in_dict = list(dict_words.keys())
    words_not_in_dict, found_here =  [], []
    
    new_lines = [] # to grab gt that will be used during training
    x = int(SC['num'] / 35) # x resembles the # of audios we will get per word
    for folder in sorted(os.listdir(SC['src_dir'])):
        folder_path = SC['src_dir'] + '/' + folder
        
        #If folder's name is in k_words, save number of instances
        if folder in k_words:
            kword_dict[folder] = x
        
        #If word no in dictionary, save it so we can add it later
        if folder not in words_in_dict:
            words_not_in_dict.append(folder)
            found_here.append(folder_path)
        
        #Grab x samples (lines) from this folder's transcript
        transcr = folder_path + '/transcript.txt'
        F = open(transcr, 'r')
        new_lines += F.readlines()[:x]
        F.close()
    
    #Shuffle lines that will be used for training
    random.shuffle(new_lines)
    
    #Determine how many samples I will need for validation
    dev_split = int(SC['splits'][1] * SC['num'])
    
    #Get train samples; save spctrgrm paths and phonemes in {train_csv}
    f = open(SC['train_csv'], 'w')
    for idx_train, item in enumerate(new_lines[dev_split:]):
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    #Get validation samples; save spctrgrm paths and phonemes in {dev_csv}
    f = open(SC['dev_csv'], 'w')
    for item in new_lines[:dev_split]:
        pt_path, old_sentence, duration = item.split('\t')
        new_sentence = []
        words = old_sentence.split(' ')
        for word in words:
            phs_seq = dict_words[word] #phonemes translation of word
            phs_seq = phs_seq.replace(' ', '_')
            new_sentence.append(phs_seq)
        
        f.write(pt_path + '\t' + ' '.join(new_sentence) + '\t' + duration)
    f.close()
    
    #Display number of instances found for each keyword
    for k,v in kword_dict.items():
        print(f"\tI found {v} instances of the word '{k}'")
    
    #Let user know if some words aren't in dictionary
    if len(words_not_in_dict) != 0:
        print(f"\tI also found {len(words_not_in_dict)} words that are not in"
               " our phonemes-dictionary. Such words are:")
        for idx, w in enumerate(words_not_in_dict):
            print(f"\t\t- {w} found here {found_here[idx]}")
    else:
        print("\tAll words are in our phonemes-dictionary. No manual additions "
              "needed.")
        
    print(f"\tFile '{SC['train_csv'].split('/')[-1]}' has been created. It "
          f"has {idx_train+1} lines.")
    print(f"\tFile '{SC['dev_csv'].split('/')[-1]}' has been created. It "
          f"has {dev_split} lines.")
    print("Finished Processing Speech Commands...\n")
    
def dataset_create_csv(k_words, dataset):
    '''Since datasets are formatted in a different way, we have to iterate
    through each of them differently. Thus, we determine which dataset we are
    dealing with and run its respective functions. These functions check that
    all words are in the dataset's dictionary and create custom csvs where
    each word is translated to IPA phonemes.'''
    
    if dataset['dataset_ID'] == 'TS':
        TS_check(k_words, dataset)
        TS_create_csv(dataset)
    elif dataset['dataset_ID'] == 'KA':
        KA_check(k_words, dataset)
        KA_create_csv(dataset)
    else:
        print("ERROR: I don't know which dataset we are dealing with. Please"
              " check your datasets' IDs.")
        sys.exit()
    # TODO
    # if TI_data['use_dataset']:
    #     TI_check(k_words, TI_data['transcript'], dicts['english']['path'])
    #     TI_create_csv(TI_data, dicts['english']['path'])
        
    # if LS_data['use_dataset']:
    #     LS_check(k_words, LS_data['transcript'], dicts['english']['path'])
    #     LS_create_csv(LS_data, dicts['english']['path'])
        
    # if SC_data['use_dataset']:
    #     SC_check_and_create_csv(k_words, SC_data, dicts['sc_engl']['path'])
    

def preprocess_data(gt_csvs_folder, k_words, datasets, train_csv, dev_csv):
    #Check if gt folder exists
    if not os.path.isdir(gt_csvs_folder):
        os.mkdir(gt_csvs_folder)        
        
    # Make sure gt folder is empty
    if len(os.listdir(gt_csvs_folder)) != 0:
        delete_contents(gt_csvs_folder)
    
    #Create csv file for each dataset that we wish to use
    for dataset in datasets:
        if dataset['use_dataset']:
            dataset_create_csv(k_words, dataset)
        
    #Merge csvs of interest in {train_csv} and {dev_csv}.
    merge_csvs(datasets, train_csv, 'train')
    merge_csvs(datasets, dev_csv, 'dev')
    