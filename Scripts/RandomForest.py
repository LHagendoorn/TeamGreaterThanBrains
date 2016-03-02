'''
Trains a random forest to the data with features per business.
Gives a classification for the test data.
'''


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from LoadData import load
import numpy

# Create train and test data

X, y = datasets.make_classification(n_samples=1000, n_features=10,
                                    n_informative=4, n_redundant=2,
                                    n_classes=4, n_clusters_per_class=1)

#data = load('input')

train_samples = 500  # Samples used for training the models

Xtrain = X[:train_samples]
Xtest = X[train_samples:]
Ytrain = y[:train_samples]
Ytest = y[train_samples:]

# Create random forest

forest = RandomForestClassifier()
forest.fit(Xtrain, Ytrain)

# Apply forest to test data
score = forest.score(Xtest, Ytest)

print "Accuracy score is %f" % score
