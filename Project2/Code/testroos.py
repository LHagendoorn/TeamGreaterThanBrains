# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 14:29:55 2016

@author: roosv_000
"""

from IO import Input

import numpy as np
import pandas as pd

df_keras=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Project2/Outputfiles/outputfile_20160528_2_keras1_throughoutputfile.csv')
df_hog= pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Project2/Outputfiles/outputfile_20160601_1_linearSVC_traindata_HOG_8_16_1_Caffe_SVC.csv')

for i in range(8061):
    
    df_keras