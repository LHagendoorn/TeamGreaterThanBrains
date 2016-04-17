# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:15:51 2016

@author: roosv_000

This script takes the histogram vectors for all businesses from the trainset and the valadation set. 
For the train set labels are known so a linear/poly SVM is trained on the train set histograms and the ground truth.
This trained SVM it then used to predict probabilties for labels/labels for the testset.
"""
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time 

#load the histogram feature vectors for the businesses in the trainset and valadation set
valhist=np.load('../Labels per photo/data_array_val_hist_9-4_OnTrainset.npy')
trainhist=np.load('../Labels per photo/data_array_train_hist_10-4_OnTrainset.npy')

#load trainbusinesses and their labels
trainlabels=pd.read_csv('../Ensembles/train.csv',sep=';')

#load the business id's for the valadation set, 
trainbus=np.load('../../trainSet.npy')

# add the labels to the train set businesses
masklabels = trainlabels[['business_id']].isin(trainbus).all(axis=1)
traintrain=trainlabels.ix[masklabels]
trainlab=traintrain.sort_values(by='business_id')

# convert numeric labels of the train set businesses to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
trainlabelsbool = trainlab['labels'].apply(to_bool)
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
Predictions_Valset=S.predict(valhist)
#Predictions_Valset=S.predict(testhist)



