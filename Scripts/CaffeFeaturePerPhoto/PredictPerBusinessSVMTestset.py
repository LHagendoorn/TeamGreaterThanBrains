# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:15:51 2016

@author: roosv_000

This script takes the histogram vectors for all businesses from the traindata and the test data. 
For the train data labels are known so a linear/poly SVM is trained on the traindata histograms and the ground truth.
This trained SVM it then used to predict probabilties for labels/labels for the testset.
"""
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import time 
from sklearn.svm import SVC

#load the histogram feature vectors for the businesses in the testset and trainset resp.
testhist=np.load('../Labels per photo/data_array_test_9-4onall.npy')
trainhist=np.load('../Labels per photo/data_array_train_9-4onall.npy')

#load trainbusinesses and their labels
trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')
trainlabels=trainlabels.sort_values(by='business_id')

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
trainlabelsbool = trainlabels['labels'].apply(to_bool)
trainlabelsbool=pd.DataFrame.as_matrix(trainlabelsbool)
print 'done intitializing data'

#TRAIN SVM
print 'Training SVM....'   
ti = time.time()
#S = OneVsRestClassifier(SVC(kernel='poly',probability=True)).fit(trainhist, trainlabelsbool)
S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainhist, trainlabelsbool)

score = S.score(trainhist,trainlabelsbool)
print time.time() - ti

#save classifier
import pickle

# now you can save it to a file
with open('svm.pkl', 'wb') as f:
    pickle.dump(S, f)


#Use the trained SVM to predict testdata
t = time.time()

print t-time.time()

Predictions_Testset=S.predict(testhist)
#Predictions_Testset=S.predict_proba(testhist)


        



