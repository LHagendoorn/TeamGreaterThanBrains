# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:15:59 2016

@author: roosv_000

Ensemble of 4 models is tested on a valadationset to determine the desired weight for each model. 
Label probalilities for the valadation businesses predicted by four different models are imported.
The F1 scores for the ensemble of models is calculated 
"""

import numpy as np
import pandas as pd

#load models, every model consist of a matrix label probabilities for all businesses in the valadation set.
model1 = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/probSTAT.csv', sep=';', header=None).values
model2 = pd.read_csv('C:/Users/roosv_000/Downloads/probSVM.csv',sep=',', header=None).values
model3 = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/probColor.csv',sep=',', header=None).values
model4 = np.load('../Labels per photo/data_array_test_9-4.onallprob.npy')

#weights for the models, if you only want to ensemble 2 methods set the third and forth value on 0
weights = [0.25, 0.65, 0.1, 0]

#classification threshold
threshold = 0.5

#load verificatie set and the full train data
veriset = np.load('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/verifSet.npy')
traindata = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/train.csv', sep=';')
trainlabelsseries = traindata['labels'].astype(str).str.split(',')
trainlabels = pd.Series.to_frame(trainlabelsseries)

#Create one matrix with the probability for each label for every business, by combining the probabilities in the models with certain weights.
ensembleprob = model1 * weights[0] + model2 * weights[1] + model3 * weights[2] + model4 * weights[3]

#Classify by converting probabilities into binaries with help of the threshold.
ensembleprob[ensembleprob > threshold] = 1
ensembleprob[ensembleprob <= threshold] = 0

#convert array ensembleprob [0 1 0 0 0 1] to list of strings ['2 6']
predList = []
for row in ensembleprob:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        labelstr = sep.join(indices)
        predList.append(labelstr)
        
        
#calculate true and false positive and false negatives


#calculate F1 score
tp=float(1)
fn=float(1)
fp=float(1)
r=tp/(tp+fn)
p=tp/(tp+fp)
F1=2*((p*r)/(p+r))
print(F1)
