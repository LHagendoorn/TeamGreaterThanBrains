from Load import *
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from itertools import chain #to flatten lists

def load_indices():
    with open('caffefeature_indices.csv', 'rb') as f:
        reader = csv.reader(f)
        indices = list(reader)
        indices = list(chain.from_iterable(indices))
        return [int(x) for x in indices]

#
# SELECT INDICES OF NON-ZERO VARIANCED FEATURES
#
def select_indices():
    print 'reading in features'
    test_features = load_testdata_caffefeatures()
    train_features = load_traindata_caffefeatures()

    print 'selecting indices'
    #get indices of features that have a non-zero variance in the test data
    selector1 = VarianceThreshold()
    selector1.fit_transform(test_features)
    indices_test = selector1.get_support(indices=True)

    #get indices of features that have a non-zero variance in the train data
    selector2 = VarianceThreshold()
    selector2.fit_transform(train_features)
    indices_train = selector2.get_support(indices=True)

    #only keep indices that have variance in both test and train data
    indices = list(set(indices_test) & set(indices_train))

    #add 1 to all indices
    indices = [x+1 for x in indices]

    #save indices to csv file
    myfile = open('caffefeature_indices.csv', 'wb')
    wr = csv.writer(myfile)
    wr.writerow(indices)


#
#   SELECT FEATURES USING THE INDICES
#

def select_features(indices):

    print 'reading in files'
    df_testdata = pd.read_csv('testdata_caffefeatures.csv',header=None)
    df_traindata = pd.read_csv('traindata_caffefeatures.csv',header=None)
    df_trainset = pd.read_csv('trainset_caffefeatures.csv',header=None)
    df_validationset = pd.read_csv('validationset_caffefeatures.csv',header=None)

    print 'selecting columns'
    df_testdata = df_testdata.ix[:,indices]
    df_traindata = df_traindata.ix[:,indices]
    df_trainset = df_trainset.ix[:,indices]
    df_validationset = df_validationset.ix[:,indices]

    print 'saving to csv'
    df_testdata.to_csv('testdata_red_caffefeatures.csv', index=False, header=False)
    df_traindata.to_csv('traindata_red_caffefeatures.csv', index=False, header=False)
    df_trainset.to_csv('trainset_red_caffefeatures.csv', index=False, header=False)
    df_validationset.to_csv('validationset_red_caffefeatures.csv', index=False, header=False)

indices = load_indices()
select_features(indices)




