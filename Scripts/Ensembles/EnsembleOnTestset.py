# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:15:59 2016

@author: roosv_000

Ensemble of 4 models. Label probalilities for the test businesses predicted by four different models are imported. 
By setting weights for each model one submission file is made.
"""
import numpy as np
import pandas as pd

#load models, every model consist of a matrix label probabilities for all businesses in the test set.
model1 = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/probSTAT.csv', sep=';', header=None).values
model2 = pd.read_csv('C:/Users/roosv_000/Downloads/probSVM.csv',sep=',', header=None).values
model3 = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/probColor.csv',sep=',', header=None).values
model4 = np.load('../Labels per photo/data_array_test_9-4.onallprob.npy')

#import the example submission file for its stucture
submit = pd.read_csv('C:/Users/roosv_000/Documents/TeamGreaterThanBrains/Scripts/Ensembles/SubmissionFormat.csv',sep=',')

#weights for the models, if you only want to ensemble 2 methods set the third and forth value on 0
weights = [0, 0.7, 0, 0.3]

# Set a filename for the submission file
filename = 'Ensembletest_MeanSVM_perphotoSVM_7_3.csv'

#classification threshold
threshold = 0.5

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

# Put labels(predList) in de submissionfile colomn named 'labels'
submit['labels' ] = predList

#save in csv file
submit.to_csv(filename,index=False)