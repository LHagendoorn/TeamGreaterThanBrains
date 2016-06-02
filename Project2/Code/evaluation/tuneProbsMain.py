from tune import tune_submissionfile
import os

#get current directory
dir = os.path.dirname(__file__)
file = os.path.join(dir,'RF_testset.csv')

tune_submissionfile(file, 22)