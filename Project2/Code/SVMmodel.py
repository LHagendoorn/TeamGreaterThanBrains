# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IO import Input
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from Output import *
import pickle

#Load data
x_testdata = load_testdata_caffefeatures(padded=True)
x_traindata = load_traindata_caffefeatures(padded=True)
x_trainset = load_trainset_caffefeatures(padded=True)
x_validationset = load_validationset_caffefeatures(padded=True)

y_traindata = np.asarray(load_traindata_labels())
y_trainset = np.asarray(load_trainset_labels())
y_validationset = np.asarray(load_validationset_labels())

#Train classifier
clf = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
clf.fit(x_traindata, y_traindata)

# now you can save it to a file
with open('classifierrbftraindata_padded.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('classifierrbftraindata_padded.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_testdata)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_traindata_padded_rbf.pkl')  # where to save it, usually as a .pkl

#df = pd.read_pickle(file_name)
#Write outputfile
check = predsdf
predsdf = check
to_outputfile(check,5,'rbfSVC_traindata_padded_3dec_highten')

