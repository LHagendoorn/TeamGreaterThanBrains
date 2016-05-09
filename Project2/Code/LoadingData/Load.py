'''
Class responsible for loading all types of data in our project.
@author: Diede Kemper
'''

import csv
from PIL import Image
from itertools import chain #to flatten lists

'''
This file assumes that all images are present in the directory 'images\\train' or 'images\\test'

This file assumes to be in the same directory as the following files:
testdata_filenames
testdata_indices

traindata_filenames
traindata_indices
traindata_labels

trainset_filenames
trainset_indices
trainset_labels

validationset_filenames
validationset_indices
validationset_labels

#TODO add features csv files
'''

'''Getting a specific image'''

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

'''Loading of caffe features'''

def load_testdata_caffefeatures():
    # TODO Implement this function
    pass

def load_traindata_caffefeatures():
    # TODO Implement this function
    pass

def load_trainset_caffefeatures():
    # TODO Implement this function
    pass

def load_validationset_caffefeatures():
    # TODO Implement this function
    pass

