# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 19:25:13 2016

@author: Laurens
"""
import numpy;
from PIL import Image;

def averageColor(image):
    colMean = [None, None, None];
    colSd = [None, None, None];
    for channel in range(3):
        pixels = image.getdata(band=channel);
        values = [];
        for pixel in pixels:
            values.append(pixel);
        arr = numpy.array(values);
        colMean[channel] = numpy.mean(arr, axis=0);
        colSd[channel] = numpy.std(arr,axis=0);
    return [colMean,colSd];
    
print averageColor(Image.open("test.jpg"));