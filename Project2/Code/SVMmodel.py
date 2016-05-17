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
import pandas as pd
from IO import Output
import pickle

#Load data
x_testdata = Input.load_testdata_caffefeatures()
x_traindata = Input.load_traindata_caffefeatures()
x_trainset = Input.load_trainset_caffefeatures()
x_validationset = Input.load_validationset_caffefeatures()

y_traindata = np.asarray(Input.load_traindata_labels())
y_trainset = np.asarray(Input.load_trainset_labels())
y_validationset = np.asarray(Input.load_validationset_labels())

#Train classifier
clf = OneVsRestClassifier(SVC(kernel='poly', probability=True))
clf.fit(x_traindata, y_traindata)

# now you can save it to a file
with open('classifierpolytraindata.pkl', 'wb') as f:
    pickle.dump(clf, f)

## and later you can load it
with open('classifierpolytraindata.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#Make predictions
preds = clf.predict_proba(x_testdata)
predsdf = pd.DataFrame(preds)
predsdf.to_pickle('predictions_traindata.pkl')  # where to save it, usually as a .pkl

#df = pd.read_pickle(file_name)
#Write outputfile
check = predsdf
Output.to_outputfile(check,5,'polySVC_traindata_2dec_highten3')

