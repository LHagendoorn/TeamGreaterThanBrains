# -*- coding: utf-8 -*-
"""
SVM for the HOG features.

"""
from IO import Input, Output
#from Input import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
import pandas as pd
#from Output import *
import pickle
#import xgboost as xgb

print('loading data')
#Load data
x_traindata = pd.read_csv('HOG_features_train_8_16_1.csv', sep=',', header=None).values
x_testdata = pd.read_csv('HOG_features_test_8_16_1.csv', sep=',', header=None).values

#load classification
y_traindata = np.asarray(Input.load_traindata_labels())

print('training classifier')
#Train classifier
clf = OneVsRestClassifier(SVC(kernel='poly', probability=True))
clf.fit(x_traindata, y_traindata)

# now you can save it to a file
with open('classifierpolytraindata_HOG_8_16_1.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('classifierpolytraindata_HOG_8_16_1.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_testdata)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_testset_HOG_8_16_1_poly_SVC.pkl')  # where to save it, usually as a .pkl

predsdf = pd.read_pickle('predictions_testset_HOG_8_16_1_poly_SVC.pkl')
#Write outputfile
check = predsdf
predsdf = check
Output.to_outputfile(check,1,'polySVC_traindata_HOG_8_16_1_SVC')

