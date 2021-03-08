'''/**************************************************************************
    File: TI_audios2spctrgrms_all.py
    Author: Mario Esparza
    Date: 03/01/2021
    
    Iterate through all .TXT files in 'TRAIN' or 'TEST' of TIMIT's corpus.
    Grab text from each .txt file; determine respective .wav file; determine
    audios duration in miliseconds; calculate (and save) spectrogram; and
    generate a transcript with three columns: path of spectrogram, text said
    in such, and audio's duration.
    
***************************************************************************''' 
from gtts import gTTS
import os
import sys

#NOTE: gTTS used to have 'es-es' Spain and 'es-us' USA as languages
#NOTE: if num of words >= 1900, you might get a "too many requests" error

def get_vocab_aolme(aolme_folder, vocabulary):
    '''Append words of AOLME that are not in {vocabulary}'''
    #Get english transcripts from aolme folder
    transcripts = []
    for folder in sorted(os.listdir(aolme_folder)):
        if not folder.endswith('Spanish'):
            folder_path = aolme_folder + '/' + folder
            for file in sorted(os.listdir(folder_path), reverse=True):
                if file.endswith('.csv'):
                    file_path = folder_path + '/' + file
                    transcripts.append(file_path)
                    break
    
    #Go through transcripts and grab those words that weren't in kaggle
    for transcript in transcripts:
        f = open(transcript, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            text = line.strip().split(',')[-1]
            for word in text.split(' '):
                if word not in vocabulary:
                    vocabulary.append(word)
                    
        f.close()
        
    return vocabulary

def get_vocab_timit(transcript, Del, vocabulary):
    '''Append words of TIMIT that are not in {vocabulary}'''
    F = open(transcript, 'r')
    for line in F:
        _, gt = line.strip().split(Del)
        words = gt.split(' ')
        for word in words:
            if word not in vocabulary:
                vocabulary.append(word)
                
    F.close()
    return vocabulary

def check_folder(this_folder):
    #Make sure {this_folder} exists. If it does, ask if ok to continue
    if not os.path.isdir(this_folder):
        os.mkdir(this_folder)
        
    if len(os.listdir(this_folder)) != 0:
        print(f"\n{this_folder} isn't empty, c for continue or q for quit?")
        if input().lower() != 'c':
            sys.exit()

timit_transcript = '/media/mario/audios/TI/transcript.txt'
aolme_folder = '/media/mario/audios/aolme_orig'
sr = 16000 #desired sample rate
lang = 'es'
folder_mp3 = '/media/mario/audios/spanglish/gtts_mp3'
#MP3's transcript with two columns: audio_path and spoken_word
transcript_mp3 = folder_mp3 + '/transcript.txt'

#Get vocabulary from Keywords-Only-TIMIT and AOLME
vocabulary = []
vocabulary = get_vocab_timit(timit_transcript, '\t', vocabulary)
vocabulary = get_vocab_aolme(aolme_folder, vocabulary)
print(f"\nVocabulary is composed of {len(vocabulary)} words")

#Make sure folder exists. If it does, ask if ok to continue
check_folder(folder_mp3)

#Remove words that are less than 3 chars (resulting audios might be too short)
words_greater_than_3 = []
for word in vocabulary:
    if len(word) >= 3:
        words_greater_than_3.append(word)

words_greater_than_3 = sorted(words_greater_than_3)
print(f"\nFrom such, {len(words_greater_than_3)} have 3 or more characters")
del vocabulary #To free memory in case it is too big

#Use {lang} to create audio of each word in {words_greater_than_3}
F = open(transcript_mp3, 'w')
for idx, word in enumerate(words_greater_than_3):
    tts = gTTS(word, lang=lang)
    audio_path = folder_mp3 + '/' + word + '.mp3'
    tts.save(audio_path)
    F.write(audio_path + '\t' + word + '\n')
    
    if idx%200 == 0:
        print("I have created {idx+1} audios so far...")
        
F.close()
print(f"\nI have created {idx+1} audios and saved them here {folder_mp3}. "
      f"Path to transcript is the following: {transcript_mp3}")
