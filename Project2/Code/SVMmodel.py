# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Input import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
from Output import *
import pickle
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
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
clf.fit(x_trainset, y_trainset)

# now you can save it to a file
with open('classifierlineartrainset_padded_SVC.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('classifierlineartrainset_padded_SVC.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_validationset)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_validationset_padded_linear_SVC.pkl')  # where to save it, usually as a .pkl

predsdf = pd.read_pickle('predictions_validationset_padded_linear_SVC.pkl')
#Write outputfile
check = predsdf
predsdf = check
to_outputfile(check,1,'linearSVC_trainset_padded_3dec_highten_SVC')


labels = load_validationset_labels()
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels,1))
          / predictions.shape[0])