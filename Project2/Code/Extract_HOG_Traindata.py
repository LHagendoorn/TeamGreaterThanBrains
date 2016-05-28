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
import pandas as pd

#filename for the output HOG features csv file
filename = 'HOG_features_train.csv'

#load file names of all training photos 
foto_ids= Input.load_traindata_filenames()
nr_of_photos =len(foto_ids)

#intiate HOG array
HOGs= []

#loop over all photos and determine HOG features
for x in range(nr_of_photos):
    
    #get current photo in grayscale
    current_photo = foto_ids[x]
    image_color = Input.get_image_by_filename(current_photo, False)
    image = color.rgb2gray(image_color)

    #calculate the Histogram of Oriented Gradients (HOG) for current image
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1,1), visualise=False)
                        
    #save the fd vectors (1d HOGS)
    HOGs.append(fd)
    
    #print progress   
    if(x%100 == 0):                  
        print ("\r{0}".format((float(x)/nr_of_photos)*100) + '% done')


#Convert the HOGs list to a dataframe, and save it to a csv file under 'filename'
hogDF = pd.DataFrame(HOGs)
hogDF.to_csv(filename, index = False , header = False) 
print('DONE')
