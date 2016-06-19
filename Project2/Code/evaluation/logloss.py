'''

This function replicates the evaluation function of the Kaggle competition.
Note: it works only for the validation set!
Usage:

from evaluation import logloss
score = logloss.compute('path/to/submission/style/csv')

@author: Diede Kemper

Steps:
1) The actual submitted predicted probabilities are replaced with max(min(p,1-10^(-15),10^(-15))
2) The submitted probabilities are rescaled (each is divided by the sum)
3) The log loss is computed, and used as score

logloss = -1/N sum over N, sum over M, y_ij x log(p_ij)

N is the number of observations
M is the number of class labels
yij is 1 if observation i is in class j and 0 otherwise
pij is the predicted probability that observation i is in class j.

'''

import pandas as pd
import numpy as np
from IO import Input
from tune import tune_probabilities
from itertools import chain #to flatten lists
import math

'''
Compute = main function: load data, tunes probabilities and computes logloss
Path to data must be given.
'''
def compute(path_to_submission_csv, scale_parameter=None):
    df_filenames, df_data = load_data(path_to_submission_csv)
    if scale_parameter is not None:
        df_data = tune_probabilities(df_data, scale_parameter)
    return compute_logloss(df_filenames, df_data)

'''
Compute = main function: load data, tunes probabilities and computes logloss
Dataframe with data must be given.
'''
def compute2(df, scale_parameter=None):
    df_filenames = df.iloc[:,0]
    df_data = df.iloc[:,1:]
    if scale_parameter is not None:
        df_data = tune_probabilities(df_data, scale_parameter)
    return compute_logloss(df_filenames, df_data)
'''
Load data
'''
def load_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    df_filenames = df['img']
    df_data = df.drop('img', axis=1)
    return df_filenames, df_data

'''
Compute the logloss, as done by kaggle. Plus preprocessing.
'''
def compute_logloss(df_filenames, df_data):
    #STEP 1: replace values
    replacer = lambda x: max(float(min(x,0.999999999999)),0.0000000000000001)
    df_data = df_data.applymap(replacer)

    #STEP 2: rescale
    df_subsum = df_data.sum(axis=1)
    df_sum = pd.concat([df_subsum, df_subsum, df_subsum, df_subsum,df_subsum,
                        df_subsum,df_subsum, df_subsum,df_subsum, df_subsum],axis=1)
    df_sum.columns = ['c0','c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    df_data = df_data / df_sum

    #STEP 3: logloss
    #load correct validationset labels
    labels = Input.load_validationset_labels()
    df_labels = pd.get_dummies(labels) #to one-hot-encoding, DataFrame automatically
    df_labels.columns = ['c0','c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    #sort data to have same order as labels
    correct_order = Input.load_validationset_filenames()
    current_order = list(df_filenames.values)
    indices = [current_order.index(filename) for filename in correct_order]
    df_data = df_data.reindex(indices)
    df_data = df_data.reset_index() #reset index --> adds new indices, old indices become column 'index'
    df_data = df_data.drop('index', axis=1) #remove this new column 'index'
	
    #select probabilities of correct classes only
    df_sparse_probs = df_data * df_labels
    probs = df_sparse_probs.values
    probs = list(chain.from_iterable(probs)) #flatten list
    probs = filter(lambda x: x!=0,probs) #remove all zeros
	
    #apply log to them and take the average
    log_probs = [math.log(p) for p in probs]
    return -(np.mean(log_probs))
