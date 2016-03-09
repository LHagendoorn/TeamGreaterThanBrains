'''
Trains a random forest to the data with features per business.
Gives a classification for the test data.
'''

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from LoadData import load, load_features
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy

'''LOAD DATA'''

# Load train data: X
featureData = load_features('input')
Xtrain = featureData['TRAIN_F']
Xcols = Xtrain.columns.tolist() #business_id, r_mean, r_sd, g_mean, g_sd, b_mean, b_sd, imagecount, h_mean, h_sd, w_mean, w_sd

# Load train data: Y
data = load('input')
Ytrain = data['Y_TRAIN']
Ycols = Ytrain.columns.tolist() #business_id, 0, 1, 2, 3, 4, 5, 6, 7, 8

'''CREATE ARRAYS'''

#merge X and Y. Reasons: order should be the same. Labels could contain businesses that are removed during preprocessing.
trainData = pd.merge(Xtrain, Ytrain, on='business_id')

#split Xtrain and Y train into two arrays, without business_id!
del Xcols[0] #remove business_id from list
del Ycols[0] #remove business_id from list
Xtrain = trainData[Xcols].values
Ytrain = trainData[Ycols].values

'''RANDOM FOREST'''

forest = RandomForestClassifier()
forest.fit(Xtrain,Ytrain)

#accuracy on train set, 10 fold cross validation
scores = cross_validation.cross_val_score(forest, Xtrain, Ytrain, cv=10, scoring='f1_weighted')

print("Accuracy RF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''CREATE CLASSIFICATION'''

#create array from test data
XtestDF = featureData['TRAIN_F'] #'TEST_F'
Xtest = XtestDF[Xcols].values
Ypred = forest.predict(Xtest)

#############################################################################################
'''OTHER CLASSIFIERS'''
'''GAUSSIAN NB'''

#gnb = GaussianNB()

#accuracy on train set, 10 fold cross validation
#scores = cross_validation.cross_val_score(gnb, Xtrain, Ytrain, cv=10, scoring='f1_weighted')

#print("Accuracy GNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''LOGISTIC REGRESSION'''

#lr = LogisticRegression()

#accuracy on train set, 10 fold cross validation
#scores = cross_validation.cross_val_score(lr, Xtrain, Ytrain, cv=10, scoring='f1_weighted')

#print("Accuracy LR: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''SVC'''

#svc = LinearSVC()

#accuracy on train set, 10 fold cross validation
#scores = cross_validation.cross_val_score(svc, Xtrain, Ytrain, cv=10, scoring='f1_weighted')

#print("Accuracy SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

