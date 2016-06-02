import os
import numpy as np
from evaluation import logloss

#get current directory
dir = os.path.dirname(__file__)

#read in submission file with probabilities
score = logloss.compute(os.path.join(dir,'RF_clean.csv'))

print score


