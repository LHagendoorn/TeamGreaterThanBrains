'''
Class responsible for loading all types of data in our project.
@author: Diede Kemper
'''

'''
Usage:
Include: "from Load import *" to import section of code.
Now you can call the functions in this file.

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
This file assumes to be in the same directory as the following csv files:
testdata_filenames
testdata_indices
testdata_caffefeatures

traindata_filenames
traindata_indices
traindata_labels
traindata_caffefeatures

trainset_filenames
trainset_indices
trainset_labels
trainset_caffefeatures

validationset_filenames
validationset_indices
validationset_labels
validationset_caffefeatures

dummy_filenames
dummy_indices
dummy_labels
dummy_caffefeatures

'''

import csv
import pandas as pd
from PIL import Image
from itertools import chain #to flatten lists

'''Getting a specific image
Note: These methods assume that all images are present in the directories 'images\\train' and 'images\\test' '''

#filename = string of filename 'img_XXX.jpg'
#testset = boolean, true if test set, false otherwise
def get_image_by_filename(filename, testset):
    if testset:
        return Image.open('images\\test\\' + filename)
    else:
        return Image.open('images\\train\\' + filename)

#index = integer
#testset = boolean, true if test set, false otherwise
def get_image_by_index(index, testset):
    filename = 'img_' + str(index) + '.jpg'
    if testset:
        return Image.open('images\\test\\' + filename)
    else:
        return Image.open('images\\train\\' + filename)

'''Loading of image file names'''

def load_testdata_filenames():
    with open('testdata_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_traindata_filenames():
    with open('traindata_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_trainset_filenames():
    with open('trainset_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_validationset_filenames():
    with open('validationset_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

def load_dummy_filenames():
    with open('dummy_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        return list(chain.from_iterable(list(reader)))

'''Loading of image indices'''

def load_testdata_indices():
    with open('testdata_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_traindata_indices():
    with open('traindata_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_trainset_indices():
    with open('trainset_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_validationset_indices():
    with open('validationset_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_dummy_indices():
    with open('dummy_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

'''Loading of correct class labels'''

def load_traindata_labels():
    with open('traindata_labels.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_trainset_labels():
    with open('trainset_labels.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_validationset_labels():
    with open('validationset_labels.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

def load_dummy_labels():
    with open('dummy_labels.csv', 'rb') as f:
        reader = csv.reader(f)
        return [int(x) for x in list(chain.from_iterable(list(reader)))]

'''Loading of caffe features'''

def load_testdata_caffefeatures():
    df = pd.read_csv('testdata_caffefeatures.csv',header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

def load_traindata_caffefeatures():
    df = pd.read_csv('traindata_caffefeatures.csv',header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

def load_trainset_caffefeatures():
    df = pd.read_csv('trainset_caffefeatures.csv',header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

def load_validationset_caffefeatures():
    df = pd.read_csv('validationset_caffefeatures.csv',header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values

def load_dummy_caffefeatures():
    df = pd.read_csv('dummy_caffefeatures.csv',header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.values