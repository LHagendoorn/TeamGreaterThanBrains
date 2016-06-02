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
from itertools import chain #to flatten lists
import math

def compute(path_to_submission_csv):

    df = pd.read_csv(path_to_submission_csv)
    df_filenames = df['img']
    df_data = df.drop('img', axis=1)

    #STEP 1: replace values
    replacer = lambda x: max(min(x,1-10**(-15)),10**(-15))
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

    #select probabilities of correct classes only
    df_sparse_probs = df_data * df_labels
    probs = df_sparse_probs.values
    probs = list(chain.from_iterable(probs)) #flatten list
    probs = filter(lambda x: x!=0,probs) #remove all zeros

    #apply log to them and take the average
    log_probs = [math.log(p) for p in probs]
    return -(np.mean(log_probs))
