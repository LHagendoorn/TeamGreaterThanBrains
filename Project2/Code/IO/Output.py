# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:12:25 2016

@author: DanielleT
"""
from Input import *
import time
import numpy

#Use it by:
#   from Output import *


#Makes the submissionfile. Please note:
#   You need testdata_filenames in the same folder as your code
#   Predsdf = the probabilities of each class per photo in a DATAFRAME, you can use pd.DataFrame(predictions)
#   Submnumber = The number of the submission that day
#   Name = The name of the approach you have used

def to_outputfile(predsdf,submnumber,name):
    labels_testdata = load_testdata_filenames()
    predsdf[(predsdf > 0.8) & (predsdf < 0.95)] = predsdf[(predsdf > 0.8) & (predsdf < 0.95)] + 0.05
    predsdf[predsdf < 0.01] = 0.01        
    df = pd.DataFrame({ 'img' : numpy.asarray(labels_testdata),
                    'c0' : predsdf.iloc[:,0],
                    'c1' : predsdf.iloc[:,1],
                    'c2' : predsdf.iloc[:,2],
                    'c3' : predsdf.iloc[:,3],
                    'c4' : predsdf.iloc[:,4],
                    'c5' : predsdf.iloc[:,5],
                    'c6' : predsdf.iloc[:,6],
                    'c7' : predsdf.iloc[:,7],
                    'c8' : predsdf.iloc[:,8],
                    'c9' : predsdf.iloc[:,9]})
    df = df[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
    timestr = time.strftime("%Y%m%d")
    filename = 'outputfile_' + timestr + '_' + str(submnumber) + '_' + name + '.csv'
    df.to_csv(filename,float_format='%.3f',index=False)   #Maybe adjust float?
    