# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Load import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from Output import *


#Load data
x_testdata = load_testdata_caffefeatures()
x_traindata = load_traindata_caffefeatures()
x_trainset = load_trainset_caffefeatures()
x_validationset = load_validationset_caffefeatures()

y_traindata = np.asarray(load_traindata_labels())
y_trainset = np.asarray(load_trainset_labels())
y_validationset = np.asarray(load_validationset_labels())

#Train classifier
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
clf.fit(x_validationset, y_validationset)

#Make predictions
preds = clf.predict_proba(x_testdata)
predsdf = pd.DataFrame(preds)

#Write outputfile
to_outputfile(predsdf,1,'testing')

