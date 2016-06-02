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
a = pd.read_csv('outputfile_20160519_1_RF_trainset_padded.csv')
b = pd.read_csv('outputfile_20160526_1_linearSVC_trainset_padded_3dec_highten_SVC.csv')
c = pd.read_csv('outputfile_20160526_1_polySVC_trainset_padded_3dec_highten_SVC.csv')

predsdf = pd.read_pickle('predictions_validationset_padded_poly_SVC.pkl')
to_outputfile(predsdf,1,'polySVC_unadjusted_validationset', clean=True, validation=True)