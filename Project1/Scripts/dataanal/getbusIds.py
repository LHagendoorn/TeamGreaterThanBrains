# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:00:02 2016

@author: Laurens
Used to analyse how many images were removed (per business) by changing the minimum image size
"""

import pandas as pd

def subtractTheCounts(x):
    return pd.Series({'business_id': x['business_id'], 'counts': x['0_x'] - x['0_y']})

def percentify(x):
    return pd.Series({'business_id': x['business_id'], 'percentage_removed': float(x['0_y'])/x['0_x']*100})

photoToBiz = pd.read_csv('test_photo_to_biz.csv', sep=',')
smallPhotoId = pd.read_csv('tooSmall4Medium.csv')

#number of businesses before removing the too small images that have fewer than 10 photos
countsPerBiz = photoToBiz.groupby('business_id').size().reset_index()
smallBiz = countsPerBiz[countsPerBiz<10]

subtractedPhotosToBizId = pd.merge(smallPhotoId, photoToBiz, on='photo_id')
countsPerBizRemoved = subtractedPhotosToBizId.groupby('business_id').size().reset_index()

mergedCounts = countsPerBiz.merge(countsPerBizRemoved, on='business_id')
leftOvers = mergedCounts.apply(subtractTheCounts, axis = 1)

percentageRemoved = mergedCounts.apply(percentify, axis = 1)

overview = percentageRemoved.merge(leftOvers, on='business_id')
overview.to_csv('mediumTooSmall.csv')