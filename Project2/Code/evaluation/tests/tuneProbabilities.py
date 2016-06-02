'''
Script that tunes the scale parameter for the random forest submission.
Just a proof of principle (or unproof...)
'''

import os
import numpy as np
from evaluation import logloss

#get current directory
dir = os.path.dirname(__file__)
file = os.path.join(dir,'SVM_clean.csv')

parameters = [3.12, 3.14, 3.16, 3.18]
scores = np.zeros(len(parameters))

#try multiple scale-parameters
for i in range(len(parameters)):
    print i
    scores[i] = logloss.compute(file, scale_parameter=parameters[i])

print scores


