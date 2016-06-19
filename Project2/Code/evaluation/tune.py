'''
This script contains two main tune functions: tune_submissionfile() and tune_probabilities()
The function tune_submissionfile() takes the path to the raw model probabilities as input, and creates a new tuned
submission file as output. As argument for scale_parameter, take the best tuned one based on the validationset. Note
that this best scale_parameter differs from model to model.
The function tune_probabilities() tunes the input probabilities by scaling them and putting them in a softmax function.
@author: Diede Kemper
'''

import os
import numpy as np
from IO import Output
from IO import Input
import pandas as pd

'''
Tunes the probabilities in the submission file using the given scale_parameter.
Note: first find the optimal scale_parameter for the validationset.
'''
def tune_submissionfile(path_to_file, scale_parameter, name, submissionnumber=1):

    #load data
    df_filenames, df_data = load_data(path_to_file)

    #tune probabilities
    df_data = tune_probabilities(df_data, scale_parameter)

    #sort data on test data order
    df_data = sort_dataframe(df_data, df_filenames)

    #create new submissionfile
    Output.to_outputfile(df_data, submissionnumber, name)

'''
Tunes the probabilities in the submission file using the cut off method.
Note: use a clean submission file as input.
'''
def tune_submissionfile_cutoff(path_to_file, name, submissionnumber=1):

    #load data
    df_filenames, df_data = load_data(path_to_file)

    #sort data on test data order
    df_data = sort_dataframe(df_data, df_filenames)

    #create new submissionfile
    Output.to_outputfile(df_data, submissionnumber, name, clean=False) #not clean!

'''
Load data
'''
def load_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    df_filenames = df['img']
    df_data = df.drop('img', axis=1)
    return df_filenames, df_data

'''
Tunes the probabilities by scaling them and putting them in a softmax function
'''
def tune_probabilities(df_data, scale_parameter):
    #remove extreme values and zeros
    replacer = lambda x: max(float(min(x,0.999999999999)),0.0000000000000001)
    df_data = df_data.applymap(replacer)

    #scale values according to the scale_parameter
    scaler = lambda x: x*scale_parameter
    df_data = df_data.applymap(scaler)

    #apply softmax
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    return df_data.apply(softmax,axis=1)

'''
Sorts the data in the dataframe to have the same order as the testdata
df_data = the data that should be sorted
df_filenames = the current order of filenames of the data
'''
def sort_dataframe(df_data, df_filenames):

    correct_order = Input.load_testdata_filenames()
    current_order = list(df_filenames.values)
    indices = [current_order.index(filename) for filename in correct_order]
    df_data = df_data.reindex(indices)
    df_data = df_data.reset_index() #reset index --> adds new indices, old indices become column 'index'
    return df_data.drop('index', axis=1) #remove this new column 'index'
