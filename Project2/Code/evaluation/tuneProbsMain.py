'''
Main script for tuning a submission file using a scale parameter. Saves the tuned submission file in a new csv file.
Note: first find the optimal scale parameter.
@author: Diede Kemper.
'''

from tune import tune_submissionfile
from tune import tune_submissionfile_cutoff
import os

#get current directory
dir = os.path.dirname(__file__)

file = os.path.join(dir,'outputfile_20160616_1_NN_11_testset.csv')
tune_submissionfile_cutoff(file, 'NN_11_cutoff')
tune_submissionfile(file, 6, 'NN_11_softmax')