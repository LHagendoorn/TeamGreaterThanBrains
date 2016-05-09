'''
This script was used to create the csv files containing all metadata of all sets.
@author: Diede Kemper
'''

import csv
import numpy as np
import re #for splitting strings with multiple delimiters
from itertools import chain # to flatten lists

with open('traindata_driver_imgs_list.csv', 'rb') as f:
    reader = csv.reader(f)
    reader.next()
    data_list = list(reader)

drivers = [row[0] for row in data_list]
unique_drivers = np.unique(drivers)

#count how many images per driver
counts = [drivers.count(driver) for driver in unique_drivers]

#sort drivers on number of images
driver_counts = zip(counts,unique_drivers)
sorted_dr_counts = sorted(driver_counts, key=lambda x:x[0])

#select 1/3 of the drivers, with varying number of images
validation_drivers = sorted_dr_counts[1::3] #start with the second, take one every three drivers.
train_drivers = [driver for driver in sorted_dr_counts if driver not in validation_drivers] #take all other drivers

#select only the drivers id
validation_drivers = [x[1] for x in validation_drivers]
train_drivers = [x[1] for x in train_drivers]

#select data rows for the sets
validation_data = [x for x in data_list if x[0] in validation_drivers]
train_data = [x for x in data_list if x[0] in train_drivers]

#write full data to csv files

myfile = open('train_driver_imgs_list.csv', 'wb')
wr = csv.writer(myfile)
wr.writerows(train_data)

myfile = open('validation_driver_imgs_list.csv', 'wb')
wr = csv.writer(myfile)
wr.writerows(validation_data)

#write filenames to csv

validation_filenames = [x[2] for x in validation_data]
train_filenames = [x[2] for x in train_data]
traindata_filenames = [x[2] for x in data_list]

myfile = open('validationset_filenames.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(validation_filenames)

myfile = open('trainset_filenames.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(train_filenames)

myfile = open('traindata_filenames.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(traindata_filenames)

#write indices to csv

#get filenames
validation_filenames = [x[2] for x in validation_data]
train_filenames = [x[2] for x in train_data]
traindata_filenames = [x[2] for x in data_list]

#get indices from the filenames
validation_indices = []
for filename in validation_filenames:
    validation_indices.append([int(s) for s in re.split('\_|\.',filename) if s.isdigit()])

train_indices = []
for filename in train_filenames:
    train_indices.append([int(s) for s in re.split('\_|\.',filename) if s.isdigit()])

traindata_indices = []
for filename in traindata_filenames:
    traindata_indices.append([int(s) for s in re.split('\_|\.',filename) if s.isdigit()])

#flatten list
validation_indices = list(chain.from_iterable(validation_indices))
train_indices = list(chain.from_iterable(train_indices))
traindata_indices = list(chain.from_iterable(traindata_indices))

#write to file
myfile = open('validationset_indices.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(validation_indices)

myfile = open('trainset_indices.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(train_indices)

myfile = open('traindata_indices.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(traindata_indices)

#write correct labels to csv

#get labels
validation_labels = [x[1] for x in validation_data]
train_labels = [x[1] for x in train_data]
traindata_labels = [x[1] for x in data_list]

#only select the number
validation_labels = [int(s[1]) for s in validation_labels]
train_labels = [int(s[1]) for s in train_labels]
traindata_labels = [int(s[1]) for s in traindata_labels]

#write to file
myfile = open('validationset_labels.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(validation_labels)

myfile = open('trainset_labels.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(train_labels)

myfile = open('traindata_labels.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(traindata_labels)