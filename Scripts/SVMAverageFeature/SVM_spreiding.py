# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:53:10 2016

@author: DanielleT
@author: Diede
"""

from datetime import datetime
import pandas as pd
import numpy as np
import LoadData
import CreateClassification as cc
from Aggregator import Aggregator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class SVM_spreiding:

    ''''Initialize SVM_spreiding. Loads the data and computes/loads aggregated data. '''''
    def __init__(self, loadAggData):
        print 'Initializing SVM: Loading Data....'
        self.loadAggData = loadAggData #boolean if aggregated data should be loaded

        #load general data
        generalData = LoadData.load('input')
        self.Xtrain = generalData['X_TRAIN']
        self.Ytrain = generalData['Y_TRAIN']
        self.Xtest = generalData['X_TEST']

        print 'Loading general data is done. Continue with loading caffe features.'
        print datetime.now()

        #if aggregated data needs to be generated, also load caffe features data
        if not self.loadAggData:
            #load train data
            trainFeatureData = LoadData.load_caffe_features('input', trainSet=True)
            self.trainFeatures = trainFeatureData['FEATURES']
            trainOrder = trainFeatureData['ORDER']

            print 'Loading caffe features is done. Continue with splitting train and validation set.'
            print datetime.now()

            #############################################
            #   SPLIT TRAINSET INTO TRAIN AND VALIDATION PART
            ##############################################

            #load business ids of train and validation set
            trainSetIds = np.load('input/trainSet.npy')
            validSetIds = np.load('input/verifSet.npy')

            #merge trainOrder with trainFeatures
            allData = pd.concat([trainOrder, self.trainFeatures], axis=1)

            #create dataframes of photo indices for train and validation set
            trainPhotoIds = self.Xtrain[self.Xtrain.business_id.isin(trainSetIds)]
            validPhotoIds = self.Xtrain[self.Xtrain.business_id.isin(validSetIds)]

            #make photo_ids strings --> merging is possible
            trainPhotoIds['photo_id'].apply(str)
            validPhotoIds['photo_id'].apply(str)

            #merge allData dataframe with photo indice dataframes --> split allData in train and validation
            self.validationFeatures = pd.merge(validPhotoIds, allData, on='photo_id')
            self.trainFeatures = pd.merge(trainPhotoIds, allData, on='photo_id')

            #delete validation dictionary from memory
            del trainFeatureData
            del trainOrder
            del allData

            #############################################
            #   COMPUTE AGGREGATED DATA
            ##############################################

            #compute validation aggregated data
            self.call_aggregator('validation')

            #delete validation data from memory
            del self.validationFeatures

            print 'VALIDATION SET DONE'
            print datetime.now()

            #compute train aggregated data
            self.call_aggregator('train')

            #delete train data from memory
            del trainFeatureData
            del self.trainFeatures
            del self.trainOrder

            print 'TRAIN SET DONE. HALFWAY THERE.'
            print datetime.now()

            #load test data
            testFeatureData = LoadData.load_caffe_features('input', trainSet=False)
            self.testFeatures = testFeatureData['FEATURES']
            self.testOrder = testFeatureData['ORDER']

            #merge testOrder with testFeatures
            self.testOrder.rename(columns={0:'photo_id'},inplace=True)
            allData = pd.concat([self.testOrder, self.testFeatures], axis=1)

            #merge testFeatures with X_test on photo_ids --> business_ids are available
            self.testFeatures = pd.merge(self.Xtest, allData, on='photo_id')

            #compute test aggregated data
            self.call_aggregator('test')

            #delete test data from memory
            del testFeatureData
            del self.testFeatures
            del self.testOrder
        else:
            self.load_aggregated_data()
        print 'Initializing SVM: Loading data is done.'
        print 'KLAAR'
        print datetime.now()


    ''''Call aggregator to compute aggregated data'''''
    def call_aggregator(self, setString):
        if setString == 'train':
            print 'Initializing SVM: Computing aggregated train data...'
            agg = Aggregator(self.trainFeatures, set='Train')
            self.aggTrainData = agg.get_aggregated_data()
        elif setString == 'validation':
            print 'Initializing SVM: Computing aggregated validation data...'
            agg = Aggregator(self.validationFeatures,  set='Validation')
            self.aggValData = agg.get_aggregated_data()
        elif setString == 'test':
            print 'Initializing SVM: Computing aggregated test data...'
            agg = Aggregator(self.testFeatures,  set='Test')
            self.aggTestData = agg.get_aggregated_data()
        else:
            print 'Error call_aggregator. This type of set is unknown.'
        print 'Initializing SVM: Computating is done.'

    ''''Load aggregated data from location'''''
    def load_aggregated_data(self):
        print 'Initializing SVM: Loading aggregated data from csv files...'
        df = pd.read_csv('./input/aggDataTrain.csv')
        self.aggTrainData = df.values
        df = pd.read_csv('./input/aggDataTest.csv')
        self.aggTestData = df.values
        df = pd.read_cs('./input/aggDataValidation.csv')
        self.aggValData = df.values
        print 'Initializing SVM: Loading is done.'

    def train(self):
        print 'Training SVM...'
        self.SVM = OneVsRestClassifier(LinearSVC(random_state=0)).fit(self.aggTrainData, self.Ytrain)
        print 'Training SVM is done.'

    def predict(self):
        print 'Predicting classes...'
        self.Ypred = self.SVM.predict(self.aggTestData)
        print 'Predicting is done.'

    def predict_prob(self):
        print 'Predicting probabilities of classes...'
        self.Yprob = self.SVM.predict(self.aggTestData)
        print 'Predicting probabilities is done.'

    def createClassification(self, filename):
        cc.create(self.Ypred, self.Xtest, filename)
        print 'Classification file created'

    def createProbabilities(self, filename):
        cc.createProbFile(self.Yprob, filename)
        print 'Probabilities file created'