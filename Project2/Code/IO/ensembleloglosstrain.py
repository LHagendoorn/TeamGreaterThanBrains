# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:43:53 2016

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

#Several files.
polyc01 = pd.read_csv('../../Outputfiles_validationset/outputfile_20160605_2_poly_c01_validationset.csv')
keras = pd.read_csv('../../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
hog = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160603_1_linearSVC_valnset_HOG_8_16_1_clean.csv')
rf = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160602_1_RF.csv')

#Kerasfile.. 
keras = keras[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
order = pd.DataFrame(load_validationset_filenames())
order.columns = ['img']
keras = order.merge(keras,on='img')

#print compute2(keras, scale_parameter=None)
#print compute2(polyc01, scale_parameter=None)
#print compute2(hog, scale_parameter=None)
#print compute2(rf, scale_parameter=None)

for i in range(0,keras.shape[0]):
    kerasi = keras.iloc[i,1:]
    if kerasi.max() < 0.95:
        keras.iloc[i,1:] = (polyc01.iloc[i,1:] + 10*kerasi)/11
        
print compute2(keras, scale_parameter=None)   





#Several files.
polyc01 = pd.read_csv('../../Outputfiles/outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
keras = pd.read_csv('../../Outputfiles/outputfile_20160611_1_KERAS_submission_loss__vgg_16_10x10_r_224_c_224_folds_10_ep_10.csv')
hog = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160603_1_linearSVC_valnset_HOG_8_16_1_clean.csv')
rf = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160602_1_RF.csv')

#Kerasfile.. 
keras = keras[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
order = pd.DataFrame(load_testdata_filenames())
order.columns = ['img']
keras = order.merge(keras,on='img')

#print compute2(keras, scale_parameter=None)
#print compute2(polyc01, scale_parameter=None)
#print compute2(hog, scale_parameter=None)
#print compute2(rf, scale_parameter=None)

for i in range(0,keras.shape[0]):
    kerasi = keras.iloc[i,1:]
    if kerasi.max() < 0.95:
        keras.iloc[i,1:] = (polyc01.iloc[i,1:] + 10*kerasi)/11
        
keras.to_csv('keras_plus_poly.csv', index = False)
print compute2(keras, scale_parameter=None)   

keras1 = pd.read_csv('../../Outputfiles/outputfile_20160527_1_KERAS_submission_loss__vgg_16_2x20_r_224_c_224_folds_2_ep_20.csv')
keras2 = pd.read_csv('../../Outputfiles/outputfile_20160608_1_KERAS_submission_loss__vgg_16_3x20_r_224_c_224_folds_3_ep_20.csv')
keras3 = pd.read_csv('../../Outputfiles/outputfile_20160611_1_KERAS_submission_loss__vgg_16_10x10_r_224_c_224_folds_10_ep_10.csv')


keras4 = keras1
keras4.iloc[:,[1,4,7]] = (keras1.iloc[:,[1,4,7]] + keras2.iloc[:,[1,4,7]] + keras3.iloc[:,[1,4,7]])/3

keras4.to_csv('keras_threeonlybestclasses.csv',float_format='%.2f', index = False)




