# End-to-end Speech Recognition of Spanish and English in Collaborative Learning Environments
This work goes hand in hand with "Spanish and English Phoneme Recognition by Training on Simulated Classroom Audio Recordings of Collaborative Learning Environments" by Mario J. Esparza \([arxiv.org/abs/2202.10536](https://arxiv.org/abs/2202.10536)\). A model based on CNN-RNN networks capable of recognizing Spanish and English phonemes. After downloading and properly configuring this project, experiments can be run on Speech Commands. Work is currently being done so it can be used with LibriSpeech, TIMIT, CSS10 and Commonvoice.

## Prerequisites
- pytorch
- numpy
- matplotlib
- torchaudio

## Steps
### Step 1: Get Things Ready


### Step 2: Phonemizing Dataset's Transcripts
This project uses bootphon's [Phonemizer](https://github.com/bootphon/phonemizer) module to translate text transcripts into IPA phonemes. To do so, use the `preprocess` command. Make sure all options are separated by a white space. These are the options you **must** add to the command:
- Name of the pre-configured dataset you are trying to use
- Path to parent directory where audio files are located
- Path to a .pickle file where IPA translations will be saved as a dictionary

For example:
`phonemize speech_commands /home/user1/Downloads/speech_commands_v2 /home/user1/Desktop/speech_commands/phonemes.pickle`

All values in hyperparameters.json should be lists. "int" and "float" determine
which datatype to set those values to.

'''
Example to phonemize speech commands dataset:
<main command> <dataset to use> <path to dataset> <path where phonemes will be saved>
#phonemize speech_commands /home/javes/Downloads/speech_commands_v2 /home/javes/Desktop/speech_commands/phonemes.pickle
            

Example to preprocess speech commands dataset:
<main command> <dataset to use> <path to dataset> <path to ground truth> <path to phonemes dictionary> -n <number of words per folder>
# preprocess speech_commands /home/javes/Downloads/speech_commands_v2 /home/javes/Desktop/speech_commands /home/javes/Desktop/speech_commands/phonemes.pickle -n 35000
            
        
Example to train speech commands dataset:
<main command> <path_to_gt> <path to phonemes dictionary> <path_to_hp_json> <path_to_other_parameters_json> <path to produced files directory>
# train /home/javes/Desktop/speech_commands/gt.csv /home/javes/Desktop/speech_commands/phonemes.pickle /home/javes/Desktop/keywordRec/hyperparameters.json /home/javes/Desktop/keywordRec/otherparameters.json /home/javes/Desktop/dummy
            
'''
