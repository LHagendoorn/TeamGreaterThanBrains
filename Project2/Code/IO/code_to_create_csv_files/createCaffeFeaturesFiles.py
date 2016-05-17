'''
This script is responsible for creating the csv files for the traindata, trainset and validationset.
It selects the correct datapoints for each set and sorts them in the standard order.
@author: Diede Kemper
'''

from Load import *
import pandas as pd
import sys

print 'reading in features'

df = pd.read_csv('features_train.csv', header=None)

print 'Old dataframe'
print df.head()

#
# TRAINDATA
#

#get filenames
traindata_filenames = load_traindata_filenames()
caffefeatures_filenames = list(df[0].values)

# check whether there are files without caffefeatures
missing_filenames = list(set(traindata_filenames) - set(caffefeatures_filenames))
if not missing_filenames: #if there are no missing files
    print 'All traindata files have caffefeatures.'
else:
    print str(len(missing_filenames)) + ' traindata files do not have caffefeatures'
    sys.exit("Program execution is stopped, because not all traindata files have caffefeatures. First solve this bug!")

# sort features on traindata filenames
indices = [caffefeatures_filenames.index(filename) for filename in traindata_filenames]
df2 = df.reindex(indices)

#save features
print 'New dataframe traindata'
print df2.head()
df2.to_csv('traindata_caffefeatures.csv', index=False, header=False) #save to csv

#
# TRAIN SET
#

# select trainset features + sort on trainset filenames

#get filenames
trainset_filenames = load_trainset_filenames()

#select rows containing train set images
df2 = df2.set_index([0]) #set the filenames as the indexing column
df_trainset = df2.loc[df2.index.isin(trainset_filenames)] #select those indices corresponding to the trainset filenames

#sort trainset features on trainset filenames
filenames = list(df_trainset.index.values) #get list of the current order of the filenames
#indices = [filenames.index(filename) for filename in trainset_filenames]
df_trainset = df_trainset.reindex(trainset_filenames)

print 'trainset filenames'
print filenames[0:10]

#save features
print 'New dataframe trainset'
print df_trainset.head()
df_trainset.to_csv('trainset_caffefeatures.csv', header=False) #save to csv

#
# VALIDATION SET
#

# select validationset features + sort on validationset filenames

#get filenames
validationset_filenames = load_validationset_filenames()

#select rows containing train set images
df_validationset = df2.loc[df2.index.isin(validationset_filenames)] #select those indices corresponding to the trainset filenames

#sort trainset features on trainset filenames
filenames = list(df_validationset.index.values) #get list of the current order of the filenames
#indices = [filenames.index(filename) for filename in validationset_filenames]
df_validationset = df_validationset.reindex(validationset_filenames)

print 'validationset filenames'
print filenames[0:10]

#save features
print 'New dataframe validationset'
print df_validationset.head()
df_validationset.to_csv('validationset_caffefeatures.csv', header=False) #save to csv