# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:17:45 2016

@author: Laurens

An SVM used to classify the skin clusters
"""

from IO import Input
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
from IO import Output
import pickle
from sklearn.svm import LinearSVC

'''
Helper function to use with the grouping of the dataframe, turns 3 rows of coordinates into a single row
'''
def transformXY(coords):
    return pd.Series(np.asarray(coords).ravel())

#Load the file names of the various datasets
trainset_filenames = Input.load_trainset_filenames()
validationset_filenames = Input.load_validationset_filenames()
traindata_filenames = Input.load_traindata_filenames()
testset_filenames = Input.load_testdata_filenames()

#Load the features
feat = pd.read_csv('skinTrainFeatures.csv', index_col = 0)

#Select the features for each dataset
x_trainset = feat.ix[trainset_filenames]
x_validationset = feat.ix[validationset_filenames]  
x_testset = feat.ix[testset_filenames]  
x_traindata = feat.ix[traindata_filenames]

#Load the labels for each dataset
y_trainset = np.asarray(Input.load_trainset_labels())
y_validationset = np.asarray(Input.load_validationset_labels())
y_traindata = np.asarray(Input.load_traindata_labels())

#restructure the features so they can be used in the SVM
x_trainset = x_trainset.groupby(x_trainset.index).apply(transformXY)
x_validationset = x_validationset.groupby(x_validationset.index).apply(transformXY)
x_testset = x_testset.groupby(x_testset.index).apply(transformXY)
x_traindata = x_traindata.groupby(x_traindata.index).apply(transformXY)

#Normalise the data
df = x_traindata.iloc[:,1:]
df_norm = (df - df.mean(axis=1)) / (df.max(axis=1) - df.min(axis=1))
x_traindata = df_norm

#Train classifier
clf = OneVsRestClassifier(SVC(C=0.1,kernel='poly', probability=True))
clf.fit(x_traindata, y_traindata)

# now you can save it to a file
with open('SKINclassifierpolytrainset_SVC_c01.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('SKINclassifierlineartraindata_onevsone_padded_SVC_rs5.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_testdata)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_SKIN_poly_c01_validationset.pkl')  # where to save it, usually as a .pkl

#Write outputfile
check = predsdf
predsdf = check
Output.to_outputfile(check,1,'SKINpoly_c01_clean_validationset',clean=True, validation=True)
Output.to_outputfile(check,1,'SKINpoly_c01_testdata',clean=False, validation=False)