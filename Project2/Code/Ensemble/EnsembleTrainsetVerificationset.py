# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:51:27 2016

@author: DanielleT
"""

import csv #to read from/write to csv files
from math import ceil #to round floats to the highest integer
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly
from PIL import Image
import numpy
import time
a = pd.read_csv('../outputfile_20160605_2_poly_c01_validationset.csv')
b = pd.read_csv('submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
c = pd.read_csv('outputfile_20160602_1_linearSVC_trainset_HOG_8_16_1_Clean.csv')
d = pd.read_csv('outputfile_20160602_1_RF.csv')

a.as_matrix()
a.transpose()

adone = transform_prob_to_classification(a)
bdone = transform_prob_to_classification(b)

adone.to_csv('transformed_polyc01_val.csv',index=True)
bdone.to_csv('transformed_keras_val.csv',index=True)

def transform_prob_to_classification(probs):
    for i in probs.iloc[:,1:].transpose():
        row = probs.iloc[i,1:]
        maxi = row.idxmax(axis=0)
        row[:] = 0
        row[maxi] = 1
        probs.iloc[i,1:] = row
    return probs
        
        


#try with loglossfunction