# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:31:04 2016

Extract Histogram of Oriented Gradients (HOG) for a given image.
The resulting histogram is saved in fd as a vector

@author: roosv_000
"""

from skimage.feature import hog
from skimage import color
from IO import Input
import numpy as np
import pandas as pd

foto_ids= Input.load_traindata_filenames()

#HOGs = np.zeros((3,27648))
HOGs= []

#for x in range(foto_ids):
for x in range(5):
    
    #get current photo in grayscale
    current_photo = foto_ids[x]
    image_color = Input.get_image_by_filename(current_photo, False)
    #image_color = imread(current_photo)
    image = color.rgb2gray(image_color)

    #calculate the Histogram of Oriented Gradients (HOG) for current image
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(1,1), visualise=True)
    #save the fd vectors (1d HOGS)
    #HOGs[x] = fd
    HOGs.append(fd)
    #H=np.hstack(fd)

                        
    
   
np.concatenate( HOGs, axis=0 )
hogDF = pd.DataFrame(HOGs)
hogDF.to_csv('HOG_features_test.csv') 
