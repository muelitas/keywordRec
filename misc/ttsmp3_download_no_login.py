'''/**************************************************************************
    File: ttsmp3_download_no_login.py
    Author(s): Mario Esparza, Luis Sanchez
    Date: 02/27/2021
    
    Automatically download audios from ttsmp3.com. Words to be downloaded will
    be chosen from AOLME's and TIMIT's vocabularies. We are using 3000chars 
    free-version in this program.

    NOTE: There might be some words that throw off ttsMP3. They give error:
    "Your audio file was created but is corrupt. Please check your message for
    invalid characters". If this happens, use {avoid_words} below to add them
    to the list; this way it is avoided in future runs.
    
***************************************************************************''' 
import os
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import sys
import time

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

person = 'Mia' #['Mia', 'Miguel', 'Lupe', 'Penelope']
avoid_words = {'Mia': [], 'Miguel': [], 'Penelope': [], 'Lupe': []}
start = 0
end = 25

tts_link = 'https://ttsmp3.com/'
#Folder where audios will be saved
folder_mp3 = '/media/mario/audios/TS_eng2esp/tts_' + person
#Folder of AOLME's gt
aolme_folder = '/media/mario/audios/aolme_orig'
#Path to TIMIT's transcript
timit_transcr = '/media/mario/audios/TI/transcript.txt'

#Make sure folder exists. If it does, ask if ok to continue
check_folder(folder_mp3)

#Get and sort vocabulary from AOLME (English words) and TIMIT
vocabulary = []
vocabulary = get_vocab_aolme(aolme_folder, vocabulary)
get_vocab_timit(timit_transcr, '\t', vocabulary)
vocabulary = sorted(vocabulary)
print(f"\nVocabulary is composed of {len(vocabulary)} words")

#If there are words that cause issues in TTSMP3, remove them from vocabulary
if len(avoid_words[person]) != 0:
    for idx, word in enumerate(vocabulary):
        if word in avoid_words[person]:
            print(f"- '{word}' has been removed from vocabulary")
            vocabulary.pop(idx)
            
    print("\nAfter removing 'avoid_words', vocabulary is composed of "
          f"{len(vocabulary)} words")

#Setup Firefox's settings so that it doesn't ask for download confirmation
fp = webdriver.FirefoxProfile()
fp.set_preference("browser.download.folderList",2)
fp.set_preference("browser.download.manager.showWhenStarting",False)
fp.set_preference("browser.download.dir", folder_mp3)
fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "audio/mp3")

#Grab driver, open website and assert
driver = webdriver.Firefox(firefox_profile=fp)
driver.get(tts_link)
assert "ttsMP3" in driver.title

for word in vocabulary[start:end]:
    select = Select(driver.find_element_by_id("sprachwahl"))
    select.select_by_value(person)
    time.sleep( 1.5 ) #Give it time to select value
    text_area = driver.find_element_by_id("voicetext")
    text_area.send_keys(word)
    
    download_button = driver.find_element_by_id("downloadenbutton")
    download_button.click()
    time.sleep( 1.5 ) #Give it time to download
    text_area.clear()

driver.close() #if you want to close tab, use driver.quit() instead

#Get list of downloaded audios
mp3_audios = sorted(os.listdir(folder_mp3))

#Access MP3 download folder and rename files (add leading zero to minutes, 
# seconds, hours, day and month)
for audio in mp3_audios:
    src = folder_mp3 + '/' + audio
    digits = audio.split('_VoiceText_')[1][:-4]
    year, month, others = digits.split('-')
    day, others = others.split('_')
    hour, minute, sec = others.split(' ')
    dst = folder_mp3 + '/'
    dst += 'ttsmp3' + year + '_' + month.zfill(2) + '_' + day.zfill(2) + '_'
    dst += hour.zfill(2) + '_' + minute.zfill(2) + '_' + sec.zfill(2) + '.mp3'
    os.rename(src, dst)
    
print(f"\nAll {len(mp3_audios)} mp3 audios have been renamed (leading zeros) "
      f"have been added. Ranging from {start} to {end}.")

#Rename files in download folder (implement a custom filename/ID)
mp3_audios = sorted(os.listdir(folder_mp3))
counter = 0
for word, audio in zip(vocabulary[start:end], mp3_audios):
    src = folder_mp3 + '/' + audio
    index = str(start + counter)
    dst = folder_mp3 + '/' + index.zfill(4) + '_' + person + '_' + word + '.mp3'
    os.rename(src, dst)
    counter += 1

print(f"\nAll {len(mp3_audios)} mp3 audios have been renamed using leading "
      f"zeroes, person's name and said word. Ranging from {start} to {end}.")
