# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:24:16 2016

@author: DanielleT
"""
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


test_photos = pd.read_csv('../../downloads/input/test_photo_to_biz.csv',sep=',')
y = pd.read_csv('../Features_data/labels_train_y.csv', header=None)
x = pd.read_csv('../Features_data/feats_train_x.csv',header=None)
tr = pd.read_csv('../../downloads/input/train.csv')
#del(x['Unnamed: 0'])
#a = tr['labels'][pd.isnull(tr['labels'])==True]
#a.index[:]
#xx = x.drop(a.index)
#del(xx['Unnamed: 0'])
S = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y)
print S.score(x,y)
newx = pd.read_csv('../Features_data/feats_test_x.csv',header=None)
#del(newx[0])
predictions = S.predict(newx)   #newx[1:]
np.savetxt("pred.csv", predictions, delimiter=",")
this = pd.DataFrame(data=predictions,columns=None)  # 1st row as the column names
this.to_csv('pred.csv',index=False)


#print outputfile
bizz = test_photos.business_id.unique()
a=0
arr = [0]*2
while True:
    strlabels = ""
    for i in range(0,9):
        pred = predictions[a]
        lab = pred[i]
        if lab == 1:
            strlabels = strlabels + " " + str(i)
    currarr = [bizz[a],strlabels.lstrip()]
    arr = np.vstack((np.array(arr),np.array(currarr)))
    if bizz[a]==bizz[-1]:
        break
    a = a+1

arr[0]=['business_id','labels']

this = pd.DataFrame(data=arr[1:,:],columns=arr[0,:])  # 1st row as the column names
this.to_csv('feats_sub_rs1002.csv',index=False)