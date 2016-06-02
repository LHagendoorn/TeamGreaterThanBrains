# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:27:05 2016

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

a = pd.read_csv('keras_in_order')
b = pd.read_csv('onevsone')
c = pd.read_csv('outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
a = a.iloc[:,1:]
b = b.iloc[:,1:]

(N,M) = a.shape
for i in range(0,N):
    acat = a.iloc[i,1:]
    bcat = b.iloc[i,1:]
    ccat = c.iloc[i,1:]
    if acat.idxmax(axis=0) == ccat.idxmax(axis=0) and acat.idxmax(axis=0) == bcat.idxmax(axis=0) and acat.max > 0.9:
        indexx = acat.idxmax(axis=0)       
        a.loc[i,1:] = 0.000000001
        a.loc[i,indexx]=0.9999
 
submnumber = 2      
name = 'correspondence_SVM_Keras_OnevsAll'
timestr = time.strftime("%Y%m%d")
filename = 'outputfile_' + timestr + '_' + str(submnumber) + '_' + name + '.csv'  
a.to_csv(filename,index=False) 
#        
#43692 a + b
#46960 a + c
#55932 b + c
#79726