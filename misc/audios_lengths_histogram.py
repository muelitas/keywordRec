'''/**************************************************************************
    File: audios_lengths_histogram.py
    Author: Mario Esparza
    Date: 03/05/2021
    
    Given a dataset's transcript (with durations of each audio), iterate
    through durations and store them in a dictionary. Use this dictionary to
    plot histogram. This should give a better idea of which bucket boundaries
    to use.
    
***************************************************************************''' 
import matplotlib.pyplot as plt

dataset = 'TIMIT'
transcr_path = '/media/mario/audios/spctrgrms/clean/TI_all_train/transcript.txt'

transcr = open(transcr_path, 'r')
lines = transcr.readlines()
transcr.close()

#Get durations of audios from transcript
durations = []
for line in lines:
    durations.append(int(line.strip().split('\t')[2]))

#Specify number of bins and plot histogram
n_bins = len(durations) // 5
fig, ax = plt.subplots()  # a figure with a single Axes
ax.set_title(f"{dataset} Audios' Lengths")
ax.hist(durations, bins=n_bins)
ax.set_xlabel('Durations in Miliseconds')
ax.set_ylabel('Repetitions')