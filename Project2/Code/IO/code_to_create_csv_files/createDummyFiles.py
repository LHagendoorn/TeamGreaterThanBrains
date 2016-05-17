'''
This script creates dummy csv files, by selecting the first 10 images from the traindata and putting their features, filenames,
indices and labels in separate csv files.
@author: Diede Kemper
'''

import pandas as pd
import csv
from itertools import chain #to flatten lists

#FEATURES
df = pd.read_csv('traindata_caffefeatures.csv', header=None, nrows=10)
df.to_csv('dummy_caffefeatures.csv', index=False, header=False) #save to csv

#FILENAMES
filenames = list(df[0].values)
myfile = open('dummy_filenames.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(filenames)

#INDICES
with open('traindata_indices.csv', 'rb') as f:
    reader = csv.reader(f)
    indices = list(reader)

indices = list(chain.from_iterable(indices)) #flatten list
indices = indices[0:9]

myfile = open('dummy_indices.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(indices)

# LABELS
with open('traindata_labels.csv', 'rb') as f:
    reader = csv.reader(f)
    labels = list(reader)

labels = list(chain.from_iterable(labels)) #flatten list
labels = labels[0:9]

myfile = open('dummy_labels.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(labels)
