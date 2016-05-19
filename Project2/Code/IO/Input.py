'''
Class responsible for loading all types of data in our project.
@author: Diede Kemper
'''

'''
Usage:
Include: "from IO import Input" to use all functions as Input.function.

Five datasets:
- testdata = data without correct labels
- traindata = validationset + trainset
- validationset = 1/3rd of traindata
- trainset = 2/3rd of traindata
- dummy = 10 images to test code with

Three or four functions per set:
- load_#SET#_caffefeatures()
- load_#SET#_filenames()
- load_#SET#_indices()
- load_#SET#_labels()

Functions return either a 1d list or 2d array
'''

'''
This file assumes to be in the same directory as a folder named 'csv_files', containing:
testdata_filenames
testdata_indices
testdata_caffefeatures
testdata_caffefeatures_padded

traindata_filenames
traindata_indices
traindata_labels
traindata_caffefeatures
traindata_caffefeatures_padded

trainset_filenames
trainset_indices
trainset_labels
trainset_caffefeatures
trainset_caffefeatures_padded

validationset_filenames
validationset_indices
validationset_labels
validationset_caffefeatures
validationset_caffefeatures_padded

dummy_filenames
dummy_indices
dummy_labels
dummy_caffefeatures

'''

import csv #to read from/write to csv files
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly

#load csv directory
dir = os.path.dirname(__file__)
csv_dir = os.path.join(dir,'csv_files')

'''Loading of image file names'''

def load_testdata_filenames():
    with open(os.path.join(csv_dir,'testdata_filenames.csv'), 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_traindata_filenames():
    with open(os.path.join(csv_dir,'traindata_filenames.csv'), 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_trainset_filenames():
    with open(os.path.join(csv_dir,'trainset_filenames.csv'), 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_validationset_filenames():
    with open(os.path.join(csv_dir,'validationset_filenames.csv'), 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_dummy_filenames():
    with open(os.path.join(csv_dir,'dummy_filenames.csv'), 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

'''Loading of image indices'''

def load_testdata_indices():
    with open(os.path.join(csv_dir,'testdata_indices.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_traindata_indices():
    with open(os.path.join(csv_dir,'traindata_indices.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_trainset_indices():
    with open(os.path.join(csv_dir,'trainset_indices.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_validationset_indices():
    with open(os.path.join(csv_dir,'validationset_indices.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_dummy_indices():
    with open(os.path.join(csv_dir,'dummy_indices.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

'''Loading of correct class labels'''

def load_traindata_labels():
    with open(os.path.join(csv_dir,'traindata_labels.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_trainset_labels():
    with open(os.path.join(csv_dir,'trainset_labels.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_validationset_labels():
    with open(os.path.join(csv_dir,'validationset_labels.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_dummy_labels():
    with open(os.path.join(csv_dir,'dummy_labels.csv'), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

'''Loading of caffe features'''

def load_testdata_caffefeatures(padded=True, userows=None):
    if padded:
        filename = 'testdata_caffefeatures_padded.csv'
    else:
        filename = 'testdata_caffefeatures.csv'

    #all rows should be used
    if userows == None:
        return read_all_rows(filename)
    #a selection of rows should be used
    else:
        N = 79726 #number of rows
        return read_rows_selection(filename,N,userows)

def load_traindata_caffefeatures(padded=True, userows=None):
    if padded:
        filename = 'traindata_caffefeatures_padded.csv'
    else:
        filename = 'traindata_caffefeatures.csv'

    #all rows should be used
    if userows == None:
        return read_all_rows(filename)
    #a selection of rows should be used
    else:
        N = 22424 #number of rows
        return read_rows_selection(filename,N,userows)

def load_trainset_caffefeatures(padded=True, userows=None):
    if padded:
        filename = 'trainset_caffefeatures_padded.csv'
    else:
        filename = 'trainset_caffefeatures.csv'

    #all rows should be used
    if userows == None:
        return read_all_rows(filename)
    #a selection of rows should be used
    else:
        N = 14363 #number of rows
        return read_rows_selection(filename,N,userows)

def load_validationset_caffefeatures(padded=True, userows=None):
    if padded:
        filename = 'validationset_caffefeatures_padded.csv'
    else:
        filename = 'validationset_caffefeatures.csv'

    #all rows should be used
    if userows == None:
        return read_all_rows(filename)
    #a selection of rows should be used
    else:
        N = 8061 #number of rows
        return read_rows_selection(filename,N,userows)

def load_dummy_caffefeatures(padded=True, userows=None):
    filename = 'dummy_caffefeatures.csv'
    #all rows should be used
    if userows == None:
        return read_all_rows(filename)
    #a selection of rows should be used
    else:
        N = 10 #number of rows
        return read_rows_selection(filename,N,userows)

'''
    help functions
'''

'''Reads the filename of a csv file to read it in.
   All rows will be read.'''
def read_all_rows(filename):
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

'''Reads the filename of a csv file to read it in.
   Only the rows specified in userows will be read. N refers to the total number of rows.'''
def read_rows_selection(filename, N, userows):
    allIndices = range(N)
    skiprows = list(set(allIndices)-set(userows)) #these rows should be skipped when reading in
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None, skiprows=skiprows)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

'''

def load_dummy_caffefeatures(padded=True, featureSelectionMethod=None, Percentile=100):
    #read in caffefeatures
    filename = 'dummy_caffefeatures.csv'
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None)

    #just use all features
    if featureSelectionMethod==None | Percentile==100:
        df.drop(df.columns[0], axis=1, inplace=True)
        return df.values

    #use a percentile when using the chi2 ranking
    elif featureSelectionMethod=='chi2' & Percentile<100 & Percentile>0:
        filename = 'feature_importance_trainset_chi2.csv'
        with open(os.path.join(csv_dir,filename), 'rb') as f:
            reader = csv.reader(f)
            feature_indices = list(reader)
            #TODO: select percentile from indices. Select these columns.
        df.drop(df.columns[0], axis=1, inplace=True)
        return df.values
    #panic breaks loose
    else:
        print 'Feature selection method does not exist, or percentile is not between 0 and 100.'

'''''