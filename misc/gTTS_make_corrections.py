'''/**************************************************************************
    File: gTTS_make_corrections.py
    Author: Mario Esparza
    Date: 03/04/2021
    
    Synthesized audios from gTTS may sound "robotic" or may be mispronounced.
    Given a list of words to remove and the directory where audios are, it
    copies the ones that are not in the list. Each copied audio is transformed
    from .mp3 to .wav; from 24,000Hz sample rate to 16,000Hz. A transcript
    with two columns is also created: full wav path and word spoken in such.
    
***************************************************************************''' 
from glob import glob
import os
from os.path import join as pj #pj stands for path.join
import shutil
import subprocess as subp
import sys

def check_folder(this_dir):
    '''If {this_dir} exists, ask if okay to overwrite; otherwise, create it'''
    if not os.path.isdir(this_dir):
            os.mkdir(this_dir)    
            
    if len(os.listdir(this_dir)) != 0:
        print(f"{this_dir} isn't empty, is it okay if I overwrite it? [y/n]")
        if input().lower() != 'y':
            sys.exit()
        else:
            shutil.rmtree(this_dir)
            os.mkdir(this_dir)

data_root = '/media/mario/audios'
mp3s_dir = data_root + '/spanglish/gtts_mp3'
wavs_dir = data_root + '/spanglish/gtts_wav'
transcr_path = data_root + '/spanglish/gtts_wav/transcript.txt'
SR = 16000

words2remove = ['actual', 'advisement', 'ages', "ain't", 'airplanes',
'alcohol', 'applicability', 'algebra', 'ambiguity', 'analyticity', 'anybody',
'anymore', 'anything', 'appeared', 'area', 'artists', 'auditors', 'autographs',
'average', 'background', 'bar', 'batting', 'becalmed', 'bombs', 'bootleggers',
'bored', 'both', 'brother', 'breaking', 'called', 'camp', 'can', 'careful',
'carol', 'carried', 'cases', 'cavity', 'cellophane', 'chair', 'changing',
'checkpoint', 'cheesy', 'civilian', 'clearly', 'closing', 'clotted', 'club',
"c'mon", 'coagulating', 'codes','coding', 'collared', 'color', 'come', 
'comment', 'common', 'consume', 'control', 'convert', 'cornered', 'correct', 
'crystallizing', 'cultures', 'cursor', 'curves', 'data', 'date', 'decide', 
'decoding', 'delineating', 'denominators', 'describe', 'designing', 'desirable', 
'desires', 'director', 'divide', 'divided', 'does', 'doesnt', 'doses', 'easy',
'edition', 'either', 'electron', 'electrostatic', 'endeavored', 'enemy', 
'enjoy', 'even', 'every', 'everybody', 'everyone', 'exchanged', 'experiment', 'facilitator', 'failure', 'fair', 
'favorable', 'favorite', 'female', 'fiery', 'filling', 'final', 'flourish', 'forty', 
'forward', 'frankfurters', 'gave', 'general', 'generals', 'getting', 'girlie', 'gives', 
'governor', 'guard', 'happen', 'happened', 'happening', 'has', 'have', 'having', 'hec', 
'hectors', "he'd", "he'll", 'hello', 'help', 'hes', 'hexa', 'hexadecimal', 
'hexadecimals', 'himself', 'historians', 'holes', 'horizon','huge', 'ices', 'idea', 
'identifiable', 'individual', 'informative', 'ingredients', 'integer', 'internet', 
'interrupt', 'intervals', 'interweaving', 'irons', 'itll', 'junior', 'jupyter', 'kernel',
'kinds', 'leading', 'longer', 'looking', 'maladies', 'mario', 'material', 'miles', 'military',
'minus', 'misleading', 'mucus', 'multilingual', 'multiplying', 'must', 'nagging', 'nearly',
'new', 'nintendo', 'normal', 'note', 'notes', 'nowhere', 'ohh', 'once', 'ooo', 'opened', 'opening', 
'opposite', 'optical', 'our', 'pace', 'pages', 'palm', 'paragraph', 'parenthesis', 'parents', 
'parrots', 'part', 'participate', 'participating', 'parts', 'paste', 'patterns', 'performers',
'perpendicular', 'perform', 'persisting', 'person', 'pixel', 'picked', 'poised', 'positive',
'possibilities', 'practiced', 'present', 'pressed', 'pretending', 'produce', 'progress',
'previously', 'printing', 'prisoners', 'prompt', 'props', 'protocol', 'proven', 'refresh',
'purchasers', 'quotes', 'rare', 'raspberry', 'really', 'rearrange', 'received', 'registration',
'relief', 'represent', 'restart', 'result', 'results', 'retirement', 'rock', 'rolled', 
'rubbed', 'sailed', 'sea', 'server', 'several', 'sheik', 'shes', 'showing', 'shutter', 'shy',
'sides', 'similar', 'simple', 'sitting', 'sleepily', 'social', 'solve', 'sort', 'soysauce',
'spectrum', 'spending', 'sponges', 'start', 'statue', 'step', 'stranded', 'struck', 
'structure', 'subject', 'super', 'sure', 'suspect', 'teeth', 'tells', 'terminal', 'terms', 
'terrible', 'the', 'than', 'their', 'them', 'there', 'theres', 'these', 'they', "they'd", 
'theyre', 'three', 'through', 'today', 'tomorrow', 'total', 'trash', 'travel', 'trials', 'tries',
'turn', 'turned', 'twelve', 'understands', 'unusual', 'upmanship', 'usages', 'use', 'using',
'values',  'variable', 'variables', 'ventures', 'very', 'victim', 'video', 'videos', 'volumes', 
'welcomed', 'whatever', 'whats', 'wherever', 'widths', 'wire', 'women', "world's", 'yeah', 'write',
"you'd", 'your', 'youre', 'zeros']

#If {wavs_dir} exists, ask if okay to overwrite; otherwise, create it
check_folder(wavs_dir)

#%% Check spelling of {words_to_remove} by checking the names in {mp3s_dir}
audios_paths = {} #at the same time, grab paths for each audio
mp3_files = glob(mp3s_dir + '/**.mp3')
for file in sorted(mp3_files):
    word = file.split('/')[-1].split('.')[0]
    if word in list(audios_paths.keys()):
        print(f"This word '{word}' is repeated. Please review it.")
        sys.exit()
    
    audios_paths[word] = pj(mp3s_dir, file)
    
for word in words2remove:
    if word not in list(audios_paths.keys()):
        print(f"This word '{word}' is misspelled. Please review it.")
        sys.exit()

#%% Convert from mp3 to wav; set sample rate to {SR}; create transcript
#Iterate through each audio, convert the ones that aren't in {words2remove}
counter = 0
new_transcr = open(transcr_path, 'w')
for word, mp3_path in audios_paths.items():
    if word not in words2remove:
        #Remove aposthrophes from word
        word = word.replace("'", "")
        
        #Convert audio using ffmpeg
        wav_path = pj(wavs_dir, word + '.wav')
        cmd = "ffmpeg -hide_banner -loglevel error -i" #removes ffmpeg verbose
        cmd += f" \"{mp3_path}\" -acodec pcm_s16le -ac 1 -ar {SR} {wav_path}"
        subp.run(cmd, shell=True)
        
        #Save info in new transcript
        new_transcr.write(wav_path + '\t' + word + '\n')
        counter += 1
        
new_transcr.close()

print(f"Originally, you had {len(list(audios_paths.keys()))} audios. From "
      f" those, {len(words2remove)} were removed and {counter} were copied.")
