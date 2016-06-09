# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 10:10:41 2016

@author: roosv_000
"""
""""
import numpy as np
import pylab as plt

data = np.arange((6**3))
data.resize((7,7,4))

def get_slice(volume, orientation, index):
    orientation2slicefunc = {
        "x" : lambda ar:ar[index,:,:], 
        "y" : lambda ar:ar[:,index,:],  
        "z" : lambda ar:ar[:,:,index]
    }
    return orientation2slicefunc[orientation](volume)

plt.subplot(221)
plt.imshow(get_slice(data, "x", 10), vmin=0, vmax=64**3)

plt.subplot(222)
plt.imshow(get_slice(data, "x", 39), vmin=0, vmax=64**3)

plt.subplot(223)
plt.imshow(get_slice(data, "y", 15), vmin=0, vmax=64**3)
plt.subplot(224)
plt.imshow(get_slice(data, "z", 25), vmin=0, vmax=64**3)  

plt.show() 
"""
import numpy as np

a = np.arange(64).reshape(4,4,4); 
c=np.rot90(a)
d=np.rot90(c)

b1=a.diagonal(0, # Main diagonals of two arrays created by skipping
    0  , # across the outer(left)-most axis last and
            2) # the "middle" (row) axis first.
            
b2=a.diagonal(0, # Main diagonals of two arrays created by skipping
    0  , # across the outer(left)-most axis last and
            1) # the "middle" (row) axis first.
            
b3=a.diagonal(0, # Main diagonals of two arrays created by skipping
    2  , # across the outer(left)-most axis last and
            1) # the "middle" (row) axis first.

b4=c.diagonal(0, # Main diagonals of two arrays created by skipping
    0  , # across the outer(left)-most axis last and
            2) # the "middle" (row) axis first.
            
b5=c.diagonal(0, # Main diagonals of two arrays created by skipping
    1  , # across the outer(left)-most axis last and
            0) # the "middle" (row) axis first.
b6=d.diagonal(0, # Main diagonals of two arrays created by skipping
    1  , # across the outer(left)-most axis last and
            2) # the "middle" (row) axis first.
