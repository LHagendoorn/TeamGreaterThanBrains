# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:51:27 2016

@author: DanielleTump 

Note: This script is not run in full, but is ran by parts of it, 
depending on the ensemble needed and the files used. 
"""

import csv #to read from/write to csv files
from math import ceil #to round floats to the highest integer
import pandas as pd #to use dataframes
from itertools import chain #to flatten lists
import os #to load csv files path names correctly
from PIL import Image
import Input 
import numpy
import time

#Load outputfiles.
a = pd.read_csv('../../Outputfiles_validationset/outputfile_20160605_2_poly_c01_validationset.csv')
b = pd.read_csv('../../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
c = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160603_1_linearSVC_valnset_HOG_8_16_1_clean.csv')
d = pd.read_csv('../../Outputfiles_validationset/Clean/outputfile_20160602_1_RF.csv')

#Kerasfile.. 
b = b[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
order = pd.DataFrame(load_validationset_filenames())
order.columns = ['img']
b = order.merge(b,on='img')
probz = b

#load true labels
labels = pd.DataFrame(load_validationset_labels())

#Transform into 1s and 0s
b = transform_prob_to_classification(b)
labels_true = transform_labels_to_classification(labels,probz)

#Get wrong classifications
newdf = abs(b.iloc[:,1:] - labels_true.iloc[:,1:])
wrong = newdf.sum(axis=1)/2
df = pd.DataFrame({ 'img' : b.iloc[:,0],
                    'wrong': wrong})
                    
#Calculate misclassifications                  
b_all = pd.read_csv('../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')
b_all = order.merge(b_all,on='img')
b_all = b_all.join(labels)
b_all.columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','true_label']
b_all2 = b_all[wrong==1]
b_all2 = b_all2.sort_values(['true_label','img'], ascending=True)
b_all2.to_csv('missclassifications_sorted.csv')

#Get counts
count_not_class(c,labels_true)
count_wrong_class(c,labels_true)

#Change uncertain probabilities
os.chdir('IO')
b_all = pd.read_csv('../../Outputfiles/outputfile_20160528_2_keras1_throughoutputfile.csv')
b_all = order.merge(b_all,on='img')
model2 = pd.read_csv('../../Outputfiles/outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
model3 = c = pd.read_csv('../../Outputfiles/outputfile_20160601_1_linearSVC_traindata_HOG_8_16_1_Caffe_SVC.csv')
def create_output(keras, model2 , model3):
    for j in range(0,keras.shape[0]):
        i = keras.iloc[j,1:]
        if max(i) < 0.99:
            newrow = (25*i + model2.iloc[j,1:] + model2.iloc[j,1:] + model3.iloc[j,1:] +0.05)/28
            keras.iloc[j,1:] = newrow
    return keras
newkeras = b_all
newkeras = create_output(b_all,model2,model3)
newkeras.to_csv('test_newkeras.csv', index=False)
#to_outputfile(newkeras,1,'test2_KerasEnsemble',clean = False, validation = True)
os.chdir('..')

#Compute logloss.
compute('IO/test_newkeras.csv', scale_parameter=None)
compute('../Outputfiles_validationset/Clean/submission_loss__vgg_16_val_2x20_r_224_c_224_folds_2_ep_20_2016-06-06-14-00.csv')


#Load data
kerass = pd.read_csv('outputfile_20160608_1_KERAS_submission_loss__vgg_16_3x20_r_224_c_224_folds_3_ep_20.csv')
kerass = kerass[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
to_outputfile(b.iloc[:,1:],1,'kerass_adjusted', clean=False, validation = False)

#Load data
order = pd.DataFrame(load_testdata_filenames())
order.columns = ['img']
val_labels = load_validationset_labels()    
#load kerasfile  

os.chdir('../../Outputfiles')
keras = pd.read_csv('outputfile_20160528_2_keras1_throughoutputfile.csv')
keras = keras[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
model2 = pd.read_csv('outputfile_20160601_1_linearSVC_traindata_HOG_8_16_1_Caffe_SVC.csv')
model3 = pd.read_csv('outputfile_20160523_2_polySVC_traindata_padded_3dec_highten_c01.csv')
keras.iloc[:,1:] = (7*keras.iloc[:,1:] + model2.iloc[:,1:] + model3.iloc[:,1])/9
keras.to_csv('test_newkeras.csv', index=False)


#Create data.
os.chdir('../../Outputfiles')
keras = pd.read_csv('outputfile_20160528_2_keras1_throughoutputfile.csv')
keras = keras[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
model2 = pd.read_csv('outputfile_20160601_1_linearSVC_traindata_HOG_8_16_1_Caffe_SVC.csv')
keras = order.merge(keras,on='img')
os.chdir('../Code/IO')

keras = create_output2(keras,model2)            #Looks per class.
keras.to_csv('test_newkeras.csv', index=False)
compute('test_newkeras.csv', scale_parameter=None) 

#Ensemble of averages
submnumber = 2
name = 'ENSEMBLE_Average_of_5best_inclkeras_right'
labels_testdata = load_testdata_filenames()
df = pd.DataFrame({ 'img' : numpy.asarray(labels_testdata),
                    'c0' : (a.iloc[:,1]+ b.iloc[:,1] + c.iloc[:,1] + d.iloc[:,1]+ e.iloc[:,1])/5,
                    'c1' : (a.iloc[:,2]+ b.iloc[:,2] + c.iloc[:,2] + d.iloc[:,2]+ e.iloc[:,2])/5,
                    'c2' : (a.iloc[:,3]+ b.iloc[:,3] + c.iloc[:,3] + d.iloc[:,3]+ e.iloc[:,3])/5,
                    'c3' : (a.iloc[:,4]+ b.iloc[:,4] + c.iloc[:,4] + d.iloc[:,4]+ e.iloc[:,4])/5,
                    'c4' : (a.iloc[:,5]+ b.iloc[:,5] + c.iloc[:,5] + d.iloc[:,5]+ e.iloc[:,5])/5,
                    'c5' : (a.iloc[:,6]+ b.iloc[:,6] + c.iloc[:,6] + d.iloc[:,6]+ e.iloc[:,6])/5,
                    'c6' : (a.iloc[:,7]+ b.iloc[:,7] + c.iloc[:,7] + d.iloc[:,7]+ e.iloc[:,7])/5,
                    'c7' : (a.iloc[:,8]+ b.iloc[:,8] + c.iloc[:,8] + d.iloc[:,8]+ e.iloc[:,8])/5,
                    'c8' : (a.iloc[:,9]+ b.iloc[:,9] + c.iloc[:,9] + d.iloc[:,9]+ e.iloc[:,9])/5,
                    'c9' : (a.iloc[:,10]+ b.iloc[:,10] + c.iloc[:,10] + d.iloc[:,10]+ e.iloc[:,10])/5})
df = df[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
timestr = time.strftime("%Y%m%d")
filename = 'outputfile_' + timestr + '_' + str(submnumber) + '_' + name + '.csv'
df.to_csv(filename,float_format='%.2f',index=False)   #Maybe adjust float?
    
#Three best kerasfiles
    order = pd.DataFrame(load_testdata_filenames())
order.columns = ['img']
keras1 = pd.read_csv('outputfile_20160611_1_KERAS_submission_loss__vgg_16_10x10_r_224_c_224_folds_10_ep_10.csv')
keras1 = keras1[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras1 = order.merge(keras1,on='img')
keras2 = pd.read_csv('outputfile_20160608_1_KERAS_submission_loss__vgg_16_3x20_r_224_c_224_folds_3_ep_20.csv')
keras2 = keras2[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras2 = order.merge(keras2,on='img')
keras3 = pd.read_csv('outputfile_20160527_1_KERAS_submission_loss__vgg_16_2x20_r_224_c_224_folds_2_ep_20.csv')
keras3 = keras3[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras3 = order.merge(keras3,on='img')
keras4 = pd.read_csv('outputfile_20160527_1_KERAS_submission_loss__vgg_16_2x20_r_224_c_224_folds_2_ep_20.csv')
keras4 = keras4[['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']]
keras4 = order.merge(keras4,on='img')

keras6 = keras1
keras6.iloc[:,1:] = 0.6*keras1.iloc[:,1:] + 0.25*keras2.iloc[:,1:] + 0.15*keras3.iloc[:,1:]
keras6.to_csv('average_of_threekeras_first06sec025third015.csv', index=False)

#Adjust very high probabilities
for r in range(0,keras6.shape[0]):
        i = keras6.iloc[r,1:]  
        if i.max() > 0.99:
            ind = i.idxmax()
            i[:] = 0.00000001
            i[ind] = 0.99999
        keras6.iloc[r,1:] = i
        if r % 1000 == 0:
            print r
            





#Calculate false negatives
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

#Calculate false positieves
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

#Transform into array of zeros and ones.
def transform_prob_to_classification(probs):
    for i in probs.iloc[:,1:].transpose():
        row = probs.iloc[i,1:]
        maxi = row.idxmax(axis=0)
        row[:] = 0
        row[maxi] = 1
        probs.iloc[i,1:] = row
    return probs
 
#Transform labels into array of zeros and ones   
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



#Look at ensemble per class dependent on the false negatives and positives.
def create_output2(keras,model2):
    for j in range(0,keras.shape[0]):
        i = keras.iloc[j,1:]        
#        if (i.idxmax() == 'c7'):
#            i[[4,5,6,9]]=0.01
##            if (i.max() - i.drop(i.idxmax()).max()) > 0.2: 
##                print i 
##                print val_labels[j]
##                i[i.idxmax()] = 0.95
##                i[i < 0.6] = 0.01
##                keras.iloc[j,1:] = i
#        if (i.idxmax() == 'c0' and i.max() < 0.7):
#            #i = (i + model2.iloc[j,1:])/2
#            i[[2,7,8]] = 0.01
#            i[9] = i[9] + 0.05
#        if (i.idxmax() == 'c1'):
#            i[:] = 0.01
#            i[1] = 0.999
#        if (i.idxmax() == 'c2'):
#            i[[0,1,3,4,7,9]] = 0.01
#            #i[8] = i[8] + 0.1
#        if (i.idxmax() == 'c3'):
#            i[[2,7,8]] = 0.01
#        if (i.idxmax() == 'c4'):
#            i[[1,2,3,9]] = 0.01
#        if (i.idxmax() == 'c5'):
#            i[[0,1,2,3,4,6,7,8]]= 0.01
#        if (i.idxmax() == 'c6'):
#            i[[0,3,5,7,9]]= 0.01
#            if i[8] < 0.9:
#                i[8] = i[8] + 0.1
        if (i.idxmax() == 'c8'):
            i[[5]] = 0.01
            if ((i[4] < 0.9 ) and (i[9] < 0.9)):
                i[[4,9]] = i[[4,9]] + 0.05
        if (i.idxmax() == 'c9'):
            i[[6,7]] = 0.01
            if i[0] < 0.9:
                i[0] = i[0] + 0.1
        keras.iloc[j,1:] = i
            
    return keras






