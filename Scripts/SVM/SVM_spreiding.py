# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:53:10 2016

@author: DanielleT
@author: Diede
"""

import LoadData
import Aggregator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class SVM_spreiding:

    ''''Initialize SVM_spreiding. Loads the data and computes/loads aggregated data. '''''
    def __init__(self, loadAggPath=None):
        print 'Initializing SVM: Loading Data....'
        self.loadAggPath = loadAggPath #path to aggregated data saved in csv file

        #load general data
        generalData = LoadData.load()
        self.Xtrain = generalData['X_TRAIN']
        self.Ytrain = generalData['Y_TRAIN']
        self.Xtest = generalData['X_TEST']

        #if aggregated data needs to be generated, also load caffe features data
        if self.loadAggPath is None:
            trainFeatureData = LoadData.load_caffe_features(trainSet=True)
            testFeatureData = LoadData.load_caffe_features(trainSet=False)

            self.trainFeatures = trainFeatureData['FEATURES']
            self.trainOrder = trainFeatureData['ORDER']
            self.testFeatures = testFeatureData['FEATURES']
            self.testOrder = testFeatureData['ORDER']

        print 'Initializing SVM: Loading data is done.'
        #get aggregated data
        self.initialize_aggregated_data()

    ''''Initialize the aggregated data. Either loads it or calls an aggregator to compute it.'''''
    def initialize_aggregated_data(self):
        if(self.loadAggPath is not None):
            self.load_aggregated_data()
        else:
            self.call_aggregator()

    ''''Call aggregator to compute aggregated data'''''
    def call_aggregator(self):
        print 'Initializing SVM: Computing aggregated train data...'
        agg = Aggregator(self.Xtrain, self.trainFeatures, self.trainOrder)
        self.aggTrainData = agg.get_aggregated_data()
        print 'Initializing SVM: Computing aggregated test data...'
        agg = Aggregator(self.Xtest, self.testFeatures, self.testOrder)
        self.aggTestData = agg.get_aggregated_data()
        print 'Initializing SVM: Computations are done.'

    ''''Load aggregated data from given location'''''
    def load_aggregated_data(self):
        print 'Initializing SVM: Loading aggregated data...'
        #TODO
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
        #TODO
        print 'TODO: create classification'

    def createProbabilities(self, filename):
        #TODO
        print 'TODO: create probabilities'