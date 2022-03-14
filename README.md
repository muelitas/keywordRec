# End-to-end Speech Recognition of Spanish and English in Collaborative Learning Environments
Originally, this project was based on keyword recognition (keyword spotting), but it switched to end-to-end speech recognition. The folder src contains the procedures and model used to train a CNN-RNN network on Spanish and English (separately or jointly). This work goes hand in hand with "Spanish and English Phoneme Recognition by Training on Simulated Classroom Audio Recordings of Collaborative Learning Environments" by Mario J. Esparza. After downloading and properly configuring, experiments can be run in LibriSpeech, TIMIT, CSS10 and Speech Commands. The labels of the network are based on phonemes. We use the module Phonemizer by bootphon to "phonemize" text transcripts.

TODO: Add Thesis to Arxiv and place link here.

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