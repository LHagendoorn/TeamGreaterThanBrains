
from sklearn.feature_selection import chi2
from IO import Input
import numpy as np
import csv

print 'loading data'
X = Input.load_trainset_caffefeatures()
Y = Input.load_trainset_labels()

print 'compute chi2 values'
chi,p = chi2(X,Y)
chi = map((lambda x:  np.inf if np.isnan(x) else x), chi) #make all nans into infs
count_inf = (np.isinf(chi)).sum()
print 'number of infinities: ' + str(count_inf) + ' of ' + str(len(chi))

print 'sort features on relevance'
indices = np.argsort(chi)

print 'save feature indices to csv'
myfile = open('feature_importance_trainset_chi2.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(indices)