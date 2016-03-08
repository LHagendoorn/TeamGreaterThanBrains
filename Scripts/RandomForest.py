'''
Trains a random forest to the data with features per business.
Gives a classification for the test data.
'''


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from LoadData import load, load_features
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy

# Load train data
data = load('input')
Ytrain = data['Y_TRAIN'] #index, business_id, 0, 1, 2, 3, 4, 5, 6, 7, 8

featureData = load_features('input')
Xtrain = featureData['TRAIN_F'] #index, business_id, r_mean, r_sd, g_mean, g_sd, b_mean, b_sd, imagecount

#merge features with labels. Reasons: order should be the same. Labels could contain businesses that are removed during preprocessing.
trainData = pd.merge(Xtrain, Ytrain, on='business_id')

#split Xtrain and Y train again, hardcoded
cols = trainData.columns.tolist()
Xcols = ['r_mean', 'r_sd', 'g_mean', 'g_sd', 'b_mean', 'b_sd', 'imagecount']
Ycols = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
Xtrain = trainData[Xcols].values
Ytrain = trainData[Ycols].values

# Create random forest

forest = RandomForestClassifier()
print 'start fitting'
forest.fit(Xtrain, Ytrain)

#accuracy on train set
print 'start scoring'
score = forest.score(Xtrain,Ytrain)

print "Accuracy score is %f" % score
