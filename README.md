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
This project uses bootphon's [Phonemizer](https://github.com/bootphon/phonemizer) module to translate text transcripts into IPA phonemes. To do so, use the `phonemize` command. Make sure all options are separated by a white space. These are the options you **must** add to the command:
- Name of the pre-configured dataset you are trying to use. Currently, available options are:
  - \"speech_commands\"
- Path to parent directory where audio files are located
- Path to a .pickle file where IPA translations will be saved as a dictionary. Translations look like this:
  - TODO

For example:
`phonemize speech_commands /home/user1/Downloads/speech_commands_v2 /home/user1/Desktop/speech_commands/phonemes.pickle`

### Step 3: Process Spectrograms
Once you have the .pickle file with IPA translations, use the `preprocess` command to obtain spectrograms from the dataset's audios. This command will also produce a .csv file with two columns: the first column will have the paths to each spectrogram and the second column will have an IPA translation of what is spoken in such spectrogram. Here's a sample of how it looks like inside the .csv file:
TODO

Options in the `preprocess` command must be separated by a white space as well. These are the options it **must** include (unless specified otherwise):
- Name of the pre-configured dataset you are trying to use \(refer to previous step for list of available pre-configured datasets)\)
- Path to parent directory where audio files are located
- Path to directory where you wish to store spectrograms
- Path to .pickle file (generated in previous step)
- \(Optional\) Command `-n` allong with the number of audios that you wish to use \(if not provided, all audios will be used\)

For example:
`preprocess speech_commands /home/user1/Downloads/speech_commands_v2 /home/user1/Desktop/speech_commands /home/user1/Desktop/speech_commands/phonemes.pickle -n 35000`

### Step 4: Train!
#### Step 4.1: Prepare Parameters
There are two JSON files that must be updated before training. One is used to set model's hyper parameters (filters, layers, dropouts, etc.) which are then broken into lists of different datatypes and iterated through in a [Parameter Grid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html). These values **must** be split by a comma `,`. I have added the file \'hyperparameters_sample.json\' as a reference. The other JSON file is used to set parameters that deal with other portions of training such as early stopping, specaugment, data split-ratio, and others. These values **must not** be split by a comma `,` since they are meant to be single values per variable.

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
