
from IO import Input
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import csv

print 'loading data'
X = Input.load_trainset_caffefeatures()
Y = Input.load_trainset_labels()

print 'make RF'
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(X,Y)

print 'take importances'
importances = forest.feature_importances_

print 'sort features on relevance'
indices = np.argsort(importances)

print 'save feature indices to csv'
myfile = open('feature_importance_RF.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(indices)