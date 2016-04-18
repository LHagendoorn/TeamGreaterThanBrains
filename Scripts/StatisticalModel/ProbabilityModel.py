# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:58:30 2016

@author: roosv_000
"""


# Sample script naive benchmark that yields 0.609 public LB score WITHOUT any image information

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data, files are assumed to be in the "../input/" directory.
train = pd.read_csv('train.csv',sep=';')
submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()

#pyarray gives for all 9 labels the probality that they occur
pyarray = np.asarray(py)
ModelProb=np.tile(pyarray, (10000, 1))

#export the predictions for all labels 
str1 = ','.join(str(e) for e in pyarray)
submit['label' ] = str1
submit.to_csv('testmodelprob.csv',index=False)