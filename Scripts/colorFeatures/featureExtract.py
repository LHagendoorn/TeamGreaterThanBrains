# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 19:25:13 2016

@author: Laurens
"""
import numpy;
from PIL import Image;
import os
import pandas as pd
import gc

train_photos = pd.read_csv('C:/Users/Laurens/Documents/uni/MLP/data/train_photo_to_biz_ids.csv',sep=',')
busIds = pd.unique(train_photos.business_id.ravel())

df = pd.DataFrame(index = busIds, columns = ['r_mean','r_sd','g_mean','g_sd','b_mean','b_sd']);

count = 0;

for busId in busIds:

    photoIds = train_photos.loc[train_photos['business_id'] == busId].photo_id.ravel()
    #intialise numpy arrays
    r = [];
    g = [];
    b = [];

    for photoId in photoIds:
        path = os.path.join('C:/Users/Laurens/Documents/uni/MLP/data/','train_photos',''.join([str(photoId),'.jpg']))
        if os.path.isfile(path):
            img = Image.open(path)
        else:
            continue
            #skips the rest of this cycle, continues with next photoId
        r.extend(img.getdata(band=0));
        g.extend(img.getdata(band=1));
        b.extend(img.getdata(band=2));
        
    nr = numpy.array(r)
    ng = numpy.array(g)
    nb = numpy.array(b)
    if nr.size > 0:
        df.loc[busId] = pd.Series({'r_mean':numpy.mean(nr),'r_sd':numpy.std(nr),'g_mean':numpy.mean(ng),'g_sd':numpy.std(ng),'b_mean':numpy.mean(nb),'b_sd':numpy.mean(nb)})
    
    count+=1
    print count
    gc.collect()

df.to_csv('features.csv')


