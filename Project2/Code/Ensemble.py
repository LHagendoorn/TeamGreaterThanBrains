# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:55:57 2016

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
a = pd.read_csv('outputfile_20160523_1_polySVC_traindata_padded_3dec_highten_c05.csv')
b = pd.read_csv('outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
c = pd.read_csv('outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
d = pd.read_csv('outputfile_20160518_8_linearSVC_traindata_padded_3dec_highten.csv')
e = pd.read_csv('outputfile_20160523_1_polySVC_traindata_padded_3dec_highten_c05.csv')

submnumber = 2
name = 'ENSEMBLE_Average_of_3best'
labels_testdata = load_testdata_filenames()
df = pd.DataFrame({ 'img' : numpy.asarray(labels_testdata),
                    'c0' : (a.iloc[:,1]+ b.iloc[:,1] + c.iloc[:,1] + d.iloc[:,1]+ e.iloc[:,1])/5,
                    'c1' : (a.iloc[:,2]+ b.iloc[:,2] + c.iloc[:,2] + d.iloc[:,2]+ e.iloc[:,2])/5,
                    'c2' : (a.iloc[:,3]+ b.iloc[:,3] + c.iloc[:,3] + d.iloc[:,3]+ e.iloc[:,3])/5,
                    'c3' : (a.iloc[:,4]+ b.iloc[:,4] + c.iloc[:,4] + d.iloc[:,4]+ e.iloc[:,4])/5,
                    'c4' : (a.iloc[:,5]+ b.iloc[:,5] + c.iloc[:,5] + d.iloc[:,5]+ e.iloc[:,5])/5,
                    'c5' : (a.iloc[:,6]+ b.iloc[:,6] + c.iloc[:,6] + d.iloc[:,6]+ e.iloc[:,6])/5,
                    'c6' : (a.iloc[:,7]+ b.iloc[:,7] + c.iloc[:,7] + d.iloc[:,7]+ e.iloc[:,7])/5,
                    'c7' : (a.iloc[:,8]+ b.iloc[:,8] + c.iloc[:,8] + d.iloc[:,8]+ e.iloc[:,8])/5,
                    'c8' : (a.iloc[:,9]+ b.iloc[:,9] + c.iloc[:,9] + d.iloc[:,9]+ e.iloc[:,9])/5,
                    'c9' : (a.iloc[:,10]+ b.iloc[:,10] + c.iloc[:,10] + d.iloc[:,10]+ e.iloc[:,10])/5})
df = df[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
timestr = time.strftime("%Y%m%d")
filename = 'outputfile_' + timestr + '_' + str(submnumber) + '_' + name + '.csv'
df.to_csv(filename,float_format='%.2f',index=False)   #Maybe adjust float?
    