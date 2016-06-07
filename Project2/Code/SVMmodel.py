# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Input import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
from Output import *
import pickle
from sklearn.svm import LinearSVC
#import xgboost as xgb

#Load data
x_testdata = load_testdata_caffefeatures(padded=True)
x_traindata = load_traindata_caffefeatures(padded=True)
x_trainset = load_trainset_caffefeatures(padded=True)
x_validationset = load_validationset_caffefeatures(padded=True)

y_traindata = np.asarray(load_traindata_labels())
y_trainset = np.asarray(load_trainset_labels())
y_validationset = np.asarray(load_validationset_labels())

#Train classifier
clf = OneVsRestClassifier(SVC(C=0.1,kernel='poly', probability=True))
clf.fit(x_trainset, y_trainset)

# now you can save it to a file
with open('classifierpolytrainset_SVC_c01.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('classifierlineartraindata_onevsone_padded_SVC_rs5.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_validationset)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_poly_c01_validationset.pkl')  # where to save it, usually as a .pkl

predsdf = pd.read_pickle('predictions_testdata__onevsone_padded_linearSVC_rs5.pkl')
#Write outputfile
check = predsdf
predsdf = check
to_outputfile(check,1,'poly_c01_clean_validationset',clean=True, validation=True)
to_outputfile(check,2,'poly_c01_validationset',clean=False, validation=True)

