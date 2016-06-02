'''
The function tune_probabilities() takes the path to the raw model probabilities as input, and creates a new tuned
submission file as output. As argument for scale_parameter, take the best tuned one based on the validationset. Note
that this best scale_parameter differs from model to model.
@author: Diede Kemper
'''

import os
import numpy as np
from IO import Output
import pandas as pd

'''
Tunes the probabilities in the submission file using the given scale_parameter.
Note: first find the optimal scale_parameter for the validationset.
'''
def tune_submissionfile(path_to_file, scale_parameter, submissionnumber=1, name=None):

    #load data
    df_filenames, df_data = load_data(path_to_file)

    #tune probabilities
    df_data = tune_probabilities(df_data, scale_parameter)

    #create new name from old name
    if name == None:
        filename = os.path.basename(path_to_file)
        filename = filename[22:] #remove 'outputfile_2016XXXX_X_'
        filename = filename[:-4] #remove .csv extension
        name = filename + '_tuned'

    #create new submissionfile
    Output.to_outputfile(df_data, submissionnumber, name)

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
    replacer = lambda x: max(min(x,1-10**(-15)),10**(-15))
    df_data = df_data.applymap(replacer)

    #scale values according to the scale_parameter
    scaler = lambda x: x*scale_parameter
    df_data = df_data.applymap(scaler)

    #apply softmax
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    return df_data.apply(softmax,axis=1)