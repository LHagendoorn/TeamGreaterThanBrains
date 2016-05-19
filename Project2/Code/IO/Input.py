'''
Class responsible for loading all types of data in our project.
@author: Diede Kemper
'''

from __future__ import division #/ is float division, // is integer division and % is modulo
import csv #to read from/write to csv files
from math import ceil #to round floats to the highest integer
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly

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

def load_testdata_caffefeatures(padded=True, userows=None, featureSelectionMethod=None, Percentile=100):
    if padded:
        filename = 'testdata_caffefeatures_padded.csv'
    else:
        filename = 'testdata_caffefeatures.csv'
    N = 79726
    return load_data(filename, N, featureSelectionMethod, Percentile, userows)

def load_traindata_caffefeatures(padded=True, userows=None, featureSelectionMethod=None, Percentile=100):
    if padded:
        filename = 'traindata_caffefeatures_padded.csv'
    else:
        filename = 'traindata_caffefeatures.csv'
    N = 22424 #number of rows
    return load_data(filename, N, featureSelectionMethod, Percentile, userows)

def load_trainset_caffefeatures(padded=True, userows=None, featureSelectionMethod=None, Percentile=100):
    if padded:
        filename = 'trainset_caffefeatures_padded.csv'
    else:
        filename = 'trainset_caffefeatures.csv'
    N = 14363 #number of rows
    return load_data(filename, N, featureSelectionMethod, Percentile, userows)

def load_validationset_caffefeatures(padded=True, userows=None, featureSelectionMethod=None, Percentile=100):
    if padded:
        filename = 'validationset_caffefeatures_padded.csv'
    else:
        filename = 'validationset_caffefeatures.csv'
    N = 8061 #number of rows
    return load_data(filename, N, featureSelectionMethod, Percentile, userows)

def load_dummy_caffefeatures(padded=True, userows=None, featureSelectionMethod=None, Percentile=100):
    if padded:
        filename = 'dummy_caffefeatures_padded.csv'
    else:
        filename = 'dummy_caffefeatures.csv'
    N = 10
    return load_data(filename, N, featureSelectionMethod, Percentile, userows)

'''
    help functions
'''

def load_data(filename, N, featureSelectionMethod, percentile, userows):
    #all features should be used
    if (featureSelectionMethod==None) | (percentile==100):
        #all rows should be used
        if userows == None:
            return read_all_rows_and_cols(filename)
        #a selection of rows should be used
        else:
            return read_rows_selection(filename,N,userows)

    #a selection of features should be used
    elif (featureSelectionMethod!=None) & (percentile<100) & (percentile>0):
        #all rows should be used
        if userows == None:
            return read_cols_selection(filename, percentile, featureSelectionMethod)
        #a selection of rows should be used
        else:
            return read_rows_and_cols_selection(filename, percentile, featureSelectionMethod, N, userows)

    else:
        print 'Percentile should be between 0 and 100.'
        pass

'''Reads in a csv file by filename.
   All rows and columns will be read.'''
def read_all_rows_and_cols(filename):
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None)
    return df.values

'''Reads in a csv file by filename.
   Only the rows specified in userows will be read. N refers to the total number of rows.'''
def read_rows_selection(filename, N, userows):
    allIndices = range(N)
    skiprows = list(set(allIndices)-set(userows)) #these rows should be skipped when reading in
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None, skiprows=skiprows)
    return df.values

'''Reads in a csv file by filename.
   Reads only the percentile of features specified by the featureSelectionMethod'''
def read_cols_selection(filename, percentile, featureSelectionMethod):

    #select features
    featureIndices = get_feature_importance_list(featureSelectionMethod)
    usecols = get_percentile_from_list(featureIndices,percentile)

    #load the data
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None, usecols=usecols)
    return df.values

''' Reads in a csv file by filename.
    Reads only the given rows and the percentile of features specified by the featureSelectionMethod.'''
def read_rows_and_cols_selection(filename, percentile, featureSelectionMethod, N, userows):
    #select rows
    allIndices = range(N)
    skiprows = list(set(allIndices)-set(userows)) #these rows should be skipped when reading in

    #select features
    featureIndices = get_feature_importance_list(featureSelectionMethod)
    usecols = get_percentile_from_list(featureIndices,percentile) #these columns should be read in

    #load the data
    df = pd.read_csv(os.path.join(csv_dir,filename),header=None, skiprows=skiprows, usecols=usecols)
    return df.values

'''Returns the feature importance list for the given featureSelectionMethod'''
def get_feature_importance_list(featureSelectionMethod):
    if featureSelectionMethod == 'chi2':
        filename = 'feature_importance_trainset_chi2.csv'
    else:
        print 'featureSelectionMethod is not recognized.'

    #read in feature importance order
    with open(os.path.join(csv_dir, filename), 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

'''Returns the given percentile from the given indices'''
def get_percentile_from_list(indices,percentile):
    length = len(indices)
    nr_to_select = int(ceil((percentile/100)*length))
    return indices[0:nr_to_select]

