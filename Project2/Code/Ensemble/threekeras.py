# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:05:30 2016

@author: DanielleT
"""

import pandas as pd #to use dataframes
import csv #to read from/write to csv files
from math import ceil #to round floats to the highest integer
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly
from PIL import Image
import numpy
import time

order = pd.DataFrame(load_testdata_filenames())
order.columns = ['img']
keras1 = pd.read_csv('outputfile_20160611_1_KERAS_submission_loss__vgg_16_10x10_r_224_c_224_folds_10_ep_10.csv')
keras1 = keras1[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras1 = order.merge(keras1,on='img')
keras2 = pd.read_csv('outputfile_20160609_2_kerass_ensemble.csv')
keras2 = keras2[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras2 = order.merge(keras2,on='img')
keras3 = pd.read_csv('outputfile_20160528_1_keras1_throughoutputfile.csv')
keras3 = keras3[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras3 = order.merge(keras3,on='img')
keras4 = pd.read_csv('outputfile_20160527_1_KERAS_submission_loss__vgg_16_2x20_r_224_c_224_folds_2_ep_20.csv')
keras4 = keras4[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras4 = order.merge(keras4,on='img')

keras6 = keras1
keras6.iloc[:,1:] = keras1.iloc[:,1:] + keras2.iloc[:,1:] + keras3.iloc[:,1:] + keras4.iloc[:,1:]
keras6.iloc[:,1:] = keras6.iloc[:,1:]/4
keras6.to_csv('average_of_threekeras.csv', index=False)


keras1 = pd.read_csv('outputfile_20160527_1_KERAS_submission_loss__vgg_16_2x20_r_224_c_224_folds_2_ep_20.csv')
keras1 = keras1[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras1 = order.merge(keras1,on='img')







for r in range(0,30):#keras1.shape[0]):
        i = keras1.iloc[r,1:]  
        j = keras2.iloc[r,1:]
        k = keras3.iloc[r,1:]
        if ((i.idxmax()!=j.idxmax()) or (j.idxmax()!=k.idxmax()) or (k.idxmax()!=i.idxmax())):
            print i
            print j
            print k
