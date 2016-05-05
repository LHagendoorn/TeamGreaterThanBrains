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
submit = pd.read_csv('sample_submission.csv',sep=';')

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()
#pylist=[py[0], py[1],py[2],py[3], py[4],py[5],py[6],py[7], py[8]]

#pyarray gives for all 9 labels the probality that they occur
pyarray = np.asarray(py)
ModelProb=np.tile(pyarray, (10000, 1))

plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# predict classes that are > 0.5, 2,3,5,6,8
submit['labels'] = '2 3 5 6 8'
#submit.to_csv('naive.csv',index=False)