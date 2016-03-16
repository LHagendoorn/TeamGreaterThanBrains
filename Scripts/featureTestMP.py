# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 19:25:13 2016

@author: Laurens
"""
import numpy;
from PIL import Image;
import os
import pandas as pd
import multiprocessing as mp
import time
import math

def calcSD(xsum, xsumSq, pcount):
    return math.sqrt(abs((xsumSq/pcount)-math.pow((xsum/pcount),2)))

def getFeatures(busId):
    train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/test_photo_to_biz.csv',sep=',')
    photoIds = train_photos.loc[train_photos['business_id'] == busId].photo_id.ravel()
    #intialise numpy arrays
    #r = [];
    #g = [];
    #b = [];

    rsum = numpy.int64(0);
    gsum = numpy.int64(0);
    bsum = numpy.int64(0);        
    rsumSq = numpy.int64(0);
    gsumSq = numpy.int64(0);
    bsumSq = numpy.int64(0);         
    missCount = numpy.int64(0);
    imageWidthSum = numpy.int64(0);    
    imageHeightSum = numpy.int64(0);
    imageWidthSumSq = numpy.int64(0);
    imageHeightSumSq = numpy.int64(0);    
    
    for photoId in photoIds:
        path = os.path.join('C:/Users/Laurens/Documents/uni/MLP/data/','test_photos_reduced',''.join([str(photoId),'r.jpg']))
        pathNonReduced = os.path.join('C:/Users/Laurens/Documents/uni/MLP/data/','test_photos',''.join([str(photoId),'.jpg']))        
        if os.path.isfile(path):
            img = Image.open(path)
        else:
            missCount += 1
            continue
            #skips the rest of this cycle, continues with next photoId
        rband = numpy.array(img.getdata(band=0))
        gband = numpy.array(img.getdata(band=1))
        bband = numpy.array(img.getdata(band=2))
        rsum = rsum + numpy.sum(rband)
        gsum = gsum + numpy.sum(gband)
        bsum = bsum + numpy.sum(bband)
        rsumSq = rsumSq + numpy.sum(numpy.square(rband))
        gsumSq = gsumSq + numpy.sum(numpy.square(gband))
        bsumSq = bsumSq + numpy.sum(numpy.square(bband))
        #access the non-reduced image
        img = Image.open(pathNonReduced)
        imageWidthSum = imageWidthSum + img.size[0]
        imageHeightSum = imageHeightSum + img.size[1]
        imageWidthSumSq = imageWidthSumSq + numpy.square(img.size[0])
        imageHeightSumSq = imageHeightSumSq + numpy.square(img.size[1])
        
    pcount = (len(photoIds)-missCount) * 10000
    imgCount = len(photoIds)
    return (busId, pd.Series({'r_mean': rsum/pcount,'r_sd':calcSD(rsum, rsumSq, pcount),'g_mean': gsum/pcount,'g_sd':calcSD(gsum, gsumSq, pcount),'b_mean': bsum/pcount,'b_sd':calcSD(bsum, bsumSq, pcount),'imagecount':imgCount, 'h_mean': imageHeightSum/imgCount, 'h_sd': calcSD(imageHeightSum, imageHeightSumSq, imgCount), 'w_mean': imageWidthSum/imgCount, 'w_sd': calcSD(imageWidthSum, imageWidthSumSq, imgCount)}))

if __name__ == '__main__':
    train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/test_photo_to_biz.csv',sep=',')
    busIds = pd.unique(train_photos.business_id.ravel())
    
    #busIds = busIds[0:30]    
    
    df = pd.DataFrame(index = busIds, columns = ['r_mean','r_sd','g_mean','g_sd','b_mean','b_sd','imagecount','h_mean','h_sd','w_mean','w_sd'])
    p = mp.Pool(5, maxtasksperchild = 10)
    count = 0;
    t0 = time.time()
    for x in p.imap(getFeatures, busIds):
        count += 1
        print count
        df.loc[x[0]] = x[1]
        if count%500==0:
            df.to_csv('testFeatures.csv')
    p.close()
    p.join()
    t1 = time.time()
    print('time: ' + str(t1-t0))
    
    df.to_csv('testFeatures.csv')


