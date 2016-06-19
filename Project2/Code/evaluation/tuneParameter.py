'''
@author: Diede Kemper
Script that can be used to find the optimal scale parameter.
'''

import os
import numpy as np
from evaluation import logloss

#get current directory
dir = os.path.dirname(__file__)
file = os.path.join(dir,'outputfile_20160617_1_linearSVC_trainset_HOG_8_16_1_clean_17_6.csv')
#file = os.path.join(dir,'Forumscript_validationset_2016-06-06.csv')

parameters =  [3, 3.2, 3.4]
scores = np.zeros(len(parameters))

#try multiple scale-parameters
for i in range(len(parameters)):
    print i
    scores[i] = logloss.compute(file, scale_parameter=parameters[i])

print scores


