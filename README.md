# End-to-end Speech Recognition of Spanish and English in Collaborative Learning Environments
This work goes hand in hand with "Spanish and English Phoneme Recognition by Training on Simulated Classroom Audio Recordings of Collaborative Learning Environments" by Mario J. Esparza \([arxiv.org/abs/2202.10536](https://arxiv.org/abs/2202.10536)\). A model based on CNN-RNN networks capable of recognizing Spanish and English phonemes. After downloading and properly configuring this project, experiments can be run on Speech Commands. Work is currently being done so it can be used with LibriSpeech, TIMIT, CSS10 and Commonvoice.

## Prerequisites
I recommend creating an environment (I used conda to do so) and install the following modules and versions:
- python 3.10.0 (using conda)
- pytorch 1.11.0 (using conda)
- cudatoolkit 11.3.1 (might be installed when installing pytorch)
- numpy 1.21.2 (using conda)
- matplotlib 3.5.1 (using conda)
- torchaudio 0.11.0 (using conda)
- scikit-learn 1.0.1 (using conda)
- scipy 1.7.3 (using conda)
- phoemizer 3.0.1 (using pip)

## Steps
### Step 1: Get Things Ready
Make sure to download and untar (if needed) the dataset that you will use.

### Step 2: Phonemizing Dataset's Transcripts
This project uses bootphon's [Phonemizer](https://github.com/bootphon/phonemizer) module to translate text transcripts into IPA phonemes. To do so, use the `phonemize` command. Make sure all options are separated by a white space. These are the options you **must** add to the command:
- Name of the pre-configured dataset you are trying to use. Currently, available options are:
  - \"speech_commands\"
- Path to parent directory where audio files are located
- Path to a .pickle file where IPA translations will be saved as a dictionary. Translations look like this:
  - up -> ʌ_p
  - yes -> j_ɛ_s
  - nine -> n_aɪ_n
  - two -> t_uː

For example:
`phonemize speech_commands /home/user1/Downloads/speech_commands_v2 /home/user1/Desktop/speech_commands/phonemes.pickle`

### Step 3: Process Spectrograms
Once you have the .pickle file with IPA translations, use the `preprocess` command to obtain spectrograms from the dataset's audios. This command will also produce a .csv file with two columns: the first column will have the paths to each spectrogram and the second column will have an IPA translation of what is spoken in such spectrogram. File will be called *gt.csv*. Here are a few samples of how information is saved inside the .csv file:
- /home/user1/Desktop/speech_commands/eight_893705bb_nohash_3.pt,eɪ_t
- /home/user1/Desktop/speech_commands/tree_0d6d7360_nohash_0.pt,t_ɹ_iː

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
There are two JSON files that must be updated before training. One is used to set model's hyper parameters (filters, layers, dropouts, etc.) which are then broken into lists of different datatypes and iterated through in a [Parameter Grid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html). These values **must** be split by a comma `,`. I have added the file \'hyperparameters_sample.json\' as a reference. The other JSON file is used to set parameters that deal with other portions of training such as early stopping, specaugment, data split-ratio, etc. These values **must not** be split by a comma `,`. They are meant to be a single value per variable. I have added the file \'otherparameters_sample.json\' as a reference.

#### Step 4.2: Train The Model
Once you have the *gt.csv*, *.pickle* and two JSON files, you should be able to run the `train` command. Opions must be split by a white space. These are the options that **must** be included with the command:
- Path to *gt.csv*
- Path to *.pickle*
- Path to *hyper parameters* json
- Path to *other parameters* json
- Path to a folder where plots, checkpoints and logs will be stored

For example:
`train /home/user1/Desktop/speech_commands/gt.csv /home/user1/Desktop/speech_commands/phonemes.pickle /home/user1/Desktop/hyperparameters.json /home/user1/Desktop/otherparameters.json /home/user1/Desktop/results`

