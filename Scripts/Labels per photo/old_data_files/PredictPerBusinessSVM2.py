# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 00:15:51 2016

@author: roosv_000
"""
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import PIL 
from PIL import Image
import os
import math
from pandas import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
import time 
from sklearn import datasets

submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

PhotoBusid=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/train_photo_to_biz_ids.csv', sep=';')
UniqueBus=np.unique(PhotoBusid['business_id'])
UniqueBus=pd.DataFrame(UniqueBus)

testhist=np.load('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Labels per photo/data_array_test_9-4onall.npy')
valhist=np.load('../data_array_val_hist_9-4_OnTrainset.npy')
veribus=np.load('../../../verifSet.npy')
trainlabels=pd.read_csv('../../Ensembles/train.csv',sep=';')

masklabels = trainlabels[['business_id']].isin(veribus).all(axis=1)
vallabels=trainlabels.ix[masklabels]
#vallabels=vallabels.sort_values(by='business_id')

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
vallabelsbool = vallabels['labels'].apply(to_bool)
vallabelsbool=pd.DataFrame.as_matrix(vallabelsbool)
print 'done intitializing data'

#TRAIN SVM
print 'Training SVM....'   
ti = time.time()
S = OneVsRestClassifier(SVC(kernel='poly',probability=True)).fit(valhist, ytrue)

#S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(valhist, vallabelsbool)
#score = S.score(x,y)
print time.time() - ti

#save classifier
import pickle

# now you can save it to a file
with open('svm.pkl', 'wb') as f:
    pickle.dump(S, f)

## and later you can load it
#with open('filename.pkl', 'rb') as f:
#    clf = pickle.load(f)



#TESTDATA
t = time.time()

print t-time.time()

bla=S.predict_proba(testhist)

predList = []
for row in bla:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        ding = sep.join(indices)
        predList.append(ding)

#create dataframe object containing business_ids and list of strings

#submit['labels' ] = predList

#save in csv file
#submit.to_csv('Ensembletest_perphotoSVM2.csv',index=False)

