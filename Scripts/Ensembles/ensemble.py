# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:15:59 2016

@author: roosv_000
"""
#ensemble of 4 models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the example submission file for its stucture
submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

#load models
model1load=pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/probSTAT.csv', sep=';', header=None)
model2load= pd.read_csv('C:/Users/roosv_000/Downloads/probSVM.csv',sep=',', header=None)

#get the probability matrix from the models
model1=model1load.values
model2=model2load.values
model3=model1load.values
model4=model1load.values

#weights for the models, if you only want to ensemble 2 methods set the third and forth value on 0
weights=[0.1, 0.9, 0, 0]


#classification threshold
threshold=0.5

ensembleprob=model1*weights[0]+model2*weights[1]+model3*weights[2]+model4*weights[3]

#Classify by converting probabilities into binaries with help of the threshold.
ensembleprob[ensembleprob > threshold] = 1
ensembleprob[ensembleprob <= threshold] = 0

#convert array ensembleprob [0 1 0 0 0 1] to list of strings ['2 6']
predList = []
for row in ensembleprob:
        indices = [str(index) for index,number in enumerate(row) if number == 1.0]
        sep = " "
        ding = sep.join(indices)
        predList.append(ding)

#create dataframe object containing business_ids and list of strings

submit['labels' ] = predList

#save in csv file
submit.to_csv('Ensemble.csv',index=False)