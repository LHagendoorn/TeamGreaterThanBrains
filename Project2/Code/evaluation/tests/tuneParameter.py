'''
@author: Diede Kemper
Script that can be used to find the optimal scale parameter.
'''

import os
import numpy as np
from evaluation import logloss

#get current directory
dir = os.path.dirname(__file__)
#file = os.path.join(dir,'HOG_SVM_validationset_2016-06-02.csv')
file = os.path.join(dir,'Forumscript_validationset_2016-06-06.csv')

parameters = [1]
scores = np.zeros(len(parameters))

#try multiple scale-parameters
for i in range(len(parameters)):
    print i
    scores[i] = logloss.compute(file)#, scale_parameter=parameters[i])

print scores


