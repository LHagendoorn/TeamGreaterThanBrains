import os
from evaluation import logloss

#get current directory
dir = os.path.dirname(__file__)

#read in submission file with probabilities
score = logloss.compute(os.path.join(dir,'RandomForest.csv'))

print score