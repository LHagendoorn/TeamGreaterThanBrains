# -*- coding: utf-8 -*-
import glob
import os
import csv
from itertools import chain

def check_validationset_missing_images():
    with open('validationset_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        valset_filenames = list(chain.from_iterable(list(reader)))
    
    filenames = []
    subdirs = [x[0] for x in os.walk("test_valset")]
    for subdir in subdirs:
        for filename in glob.glob(subdir + '\*.jpg'):
            filenames.append(filename)

    for i in valset_filenames:
        count = 0
        for j in filenames:
            if i in j:
                count = count + 1
                break
        if count == 0:
            print(i)

def check_trainset_missing_images():
    with open('trainset_filenames.csv', 'rb') as f:
        reader = csv.reader(f)
        valset_filenames = list(chain.from_iterable(list(reader)))
    
    filenames = []
    subdirs = [x[0] for x in os.walk("train_valset")]
    for subdir in subdirs:
        for filename in glob.glob(subdir + '\*.jpg'):
            filenames.append(filename)

    for i in valset_filenames:
        count = 0
        for j in filenames:
            if i in j:
                count = count + 1
                break
        if count == 0:
            print(i)

#check_validationset_missing_images()

check_trainset_missing_images()