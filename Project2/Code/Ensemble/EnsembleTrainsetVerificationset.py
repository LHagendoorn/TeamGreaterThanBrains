# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:51:27 2016

@author: DanielleT
"""

import csv #to read from/write to csv files
from math import ceil #to round floats to the highest integer
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly
from PIL import Image
import numpy
import time
a = pd.read_csv('../../Outputfiles_validationset/outputfile_20160605_2_poly_c01_validationset.csv')
b = pd.read_csv('../../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
c = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160602_1_linearSVC_trainset_HOG_8_16_1_Clean.csv')
d = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160602_1_RF.csv')
b = b[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
order = pd.DataFrame(load_validationset_filenames())
order.columns = ['img']
b = order.merge(b,on='img')
probz = b
labels = load_validationset_labels()
labels = pd.DataFrame(labels)
b = transform_prob_to_classification(b)
c = transform_prob_to_classification(c)
c = b
labels_true = transform_labels_to_classification(labels,probz)

newdf = abs(b.iloc[:,1:] - labels_true.iloc[:,1:])
wrong = newdf.sum(axis=1)/2
df = pd.DataFrame({ 'img' : b.iloc[:,0],
                    'wrong': wrong})
b_all = pd.read_csv('../../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
b_all = order.merge(b_all,on='img')
b_all = b_all.join(labels)
b_all.columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','true_label']
b_all2 = b_all[wrong==1]
b_all2.to_csv('missclassifications.csv')


count_not_class(c,labels_true)
count_wrong_class(c,labels_true)

def count_not_class(datafr, labels_true):
    print "Amount of not classified (when it was that class) in the following classes"
    newdf = datafr.iloc[:,1:] - labels_true.iloc[:,1:]
    print 'c0: %d' % newdf.iloc[:,0].value_counts()[-1]
    print 'c1: %d' % newdf.iloc[:,1].value_counts()[-1]
    print 'c2: %d' % newdf.iloc[:,2].value_counts()[-1]
    print 'c3: %d' % newdf.iloc[:,3].value_counts()[-1]
    print 'c4: %d' % newdf.iloc[:,4].value_counts()[-1]
    print 'c5: %d' % newdf.iloc[:,5].value_counts()[-1]
    print 'c6: %d' % newdf.iloc[:,6].value_counts()[-1]
    print 'c7: %d' % newdf.iloc[:,7].value_counts()[-1]
    print 'c8: %d' % newdf.iloc[:,8].value_counts()[-1]
    print 'c9: %d' % newdf.iloc[:,9].value_counts()[-1]
    return

def count_wrong_class(datafr, labels_true):
    print "Amount of wrongly classified in the following classes"
    newdf = datafr.iloc[:,1:] - labels_true.iloc[:,1:]
    print 'c0: %d' % newdf.iloc[:,0].value_counts()[1]
    print 'c1: %d' % newdf.iloc[:,1].value_counts()[1]
    print 'c2: %d' % newdf.iloc[:,2].value_counts()[1]
    print 'c3: %d' % newdf.iloc[:,3].value_counts()[1]
    print 'c4: %d' % newdf.iloc[:,4].value_counts()[1]
    print 'c5: %d' % newdf.iloc[:,5].value_counts()[1]
    print 'c6: %d' % newdf.iloc[:,6].value_counts()[1]
    print 'c7: %d' % newdf.iloc[:,7].value_counts()[1]
    print 'c8: %d' % newdf.iloc[:,8].value_counts()[1]
    print 'c9: %d' % newdf.iloc[:,9].value_counts()[1]
    return


b[wrong==2]
labels[labels.iloc[:,0] == 0]


def transform_prob_to_classification(probs):
    for i in probs.iloc[:,1:].transpose():
        row = probs.iloc[i,1:]
        maxi = row.idxmax(axis=0)
        row[:] = 0
        row[maxi] = 1
        probs.iloc[i,1:] = row
    return probs
    
def transform_labels_to_classification(label,probz):
    rows = [];
    ct = 0
    label = label.transpose()
    for i in label.iloc[0,0:]:
        newrow = [0]*10;
        newrow[i] = 1;
        probz.iloc[ct,1:]=newrow
        ct = ct+1
    return probz
        


#try with loglossfunction