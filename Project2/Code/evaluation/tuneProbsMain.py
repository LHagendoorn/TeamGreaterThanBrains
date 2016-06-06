'''
Main script for tuning a submission file using a scale parameter. Saves the tuned submission file in a new csv file.
Note: first find the optimal scale parameter.
@author: Diede Kemper.
'''




from tune import tune_submissionfile
import os

#get current directory
dir = os.path.dirname(__file__)
file = os.path.join(dir,'RF_testset.csv')

tune_submissionfile(file, 22)