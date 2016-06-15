# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:17:45 2016

@author: Laurens
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

def transformXY(coords):
    return pd.Series(np.asarray(coords).ravel())

trainset_filenames = Input.load_trainset_filenames()
validationset_filenames = Input.load_validationset_filenames()
feat = pd.read_csv('skinTrainFeatures.csv', index_col = 0)
x_trainset = feat.ix[trainset_filenames]
x_validationset = feat.ix[validationset_filenames]  
y_trainset = np.asarray(Input.load_trainset_labels())
y_validationset = np.asarray(Input.load_validationset_labels())
x_trainset = np.asarray(x_trainset.groupby(x_trainset.index).apply(transformXY))
x_validationset = np.asarray(x_validationset.groupby(x_validationset.index).apply(transformXY))
#Train classifier
clf = OneVsRestClassifier(SVC(C=0.1,kernel='poly', probability=True))
clf.fit(x_trainset, y_trainset)

# now you can save it to a file
with open('SKINclassifierpolytrainset_SVC_c01.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('SKINclassifierlineartraindata_onevsone_padded_SVC_rs5.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_validationset)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_SKIN_poly_c01_validationset.pkl')  # where to save it, usually as a .pkl

#predsdf = pd.read_pickle('predictions_testdata__onevsone_padded_linearSVC_rs5.pkl')
#Write outputfile
check = predsdf
predsdf = check
Output.to_outputfile(check,1,'SKINpoly_c01_clean_validationset',clean=True, validation=True)
Output.to_outputfile(check,2,'SKINpoly_c01_validationset',clean=False, validation=True)