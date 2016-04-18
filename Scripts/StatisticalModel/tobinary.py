# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:47:46 2016

@author: roosv_000
"""
import numpy as np
import pandas as pd

train = pd.read_csv('128linearSubmission.csv',sep=',')


# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
Y = train['labels'].apply(to_bool)

ant=np.asmatrix(Y)