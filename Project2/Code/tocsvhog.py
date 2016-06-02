# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:12:26 2016

@author: roosv_000
"""

from IO import Output
import pandas as pd

predsdf = pd.read_pickle(r'C:\Users\roosv_000\Desktop\predictions_testset_HOG_8_16_1.pkl')
#Write outputfile
check = predsdf
predsdf = check
Output.to_outputfile(check,1,'linearSVC_traindata_HOG_8_16_1_No_Adjustments_to_output')

