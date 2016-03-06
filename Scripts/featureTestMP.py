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
    train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv',sep=',')
    photoIds = train_photos.loc[train_photos['business_id'] == busId].photo_id.ravel()
    #intialise numpy arrays
    #r = [];
    #g = [];
    #b = [];

    rsum = 0;
    gsum = 0;
    bsum = 0;        
    rsumSq = 0;
    gsumSq = 0;
    bsumSq = 0;         
    missCount = 0;
    
    for photoId in photoIds:
        path = os.path.join('C:/Users/Laurens/Documents/uni/MLP/data/','train_photos_reduced',''.join([str(photoId),'r.jpg']))
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
    
    pcount = (len(photoIds)-missCount) * 10000
    return (busId, pd.Series({'r_mean': rsum/pcount,'r_sd':calcSD(rsum, rsumSq, pcount),'g_mean': gsum/pcount,'g_sd':calcSD(gsum, gsumSq, pcount),'b_mean': bsum/pcount,'b_sd':calcSD(bsum, bsumSq, pcount),'imagecount':len(photoIds)}))

if __name__ == '__main__':
    train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv',sep=',')
    busIds = pd.unique(train_photos.business_id.ravel())
    
    #busIds = busIds[0:9]    
    
    df = pd.DataFrame(index = busIds, columns = ['r_mean','r_sd','g_mean','g_sd','b_mean','b_sd','imagecount']);
    p = mp.Pool(6, maxtasksperchild = 10)
    count = 0;
    t0 = time.time()
    for x in p.imap(getFeatures, busIds):
        count += 1
        print count
        df.loc[x[0]] = x[1]
    p.close()
    p.join()
    t1 = time.time()
    print('time: ' + str(t1-t0))
    
    df.to_csv('features.csv')


