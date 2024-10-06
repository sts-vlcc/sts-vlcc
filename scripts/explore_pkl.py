"""Create a script to print a pkl file."""

import pickle
import os
folder = 'datasets/SIQ2/'
# Open the pkl file in read mode
with open(os.path.join(folder,'subtitles_trimmed.pkl'), 'rb') as f:
    # Load the contents of the pkl file
    data = pickle.load(f)
    subdata = data['3vZccD8ySXc']

# Print the contents of the pkl file
print(subdata)