# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 15:23:25 2016

@author: roosv_000
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 14:54:52 2016

@author: roosv_000

This script reduces the images in the test photos folder to 100 x 100 pixels, if 
the orginal image is already smaller then 100 x 100 pixels it is put in a seperate folder.
The names of the photo files stay the same exect an 'r' is added. 

If you want to use this script do not forget to change the directories.  

"""
import PIL 
from PIL import Image
import os

# set paramaters of the desired dimensions for the foto's 
new_width=100
new_height=100
number_of_pixels = 100

#set directories
dir_test_photos='C:/Users/Laurens/Documents/uni/MLP/data/test_photos' # where the orignial test photo's are
dir_test_photos_reduced='C:/Users/Laurens/Documents/uni/MLP/data/test_photos_reduced/' #where you want to put the reduced images.
dir_test_photos_reduced_too_small='C:/Users/Laurens/Documents/uni/MLP/data/test_photos_reduced_too_small/' #where you want to put the images that were too small to reduce.


# get a list of all files in the test_photos folder, then remove the ._ files. 
all_files_list=os.listdir(dir_test_photos) 
test_photos_list=[ x for x in all_files_list if "_" not in x ]
test_photos_list=[s.replace('.jpg', '') for s in test_photos_list]

#get the number of images in the list 
number_of_images=len(test_photos_list)

# for every image do the following:
for x in range(0, number_of_images-1):
    
    #import the image
    scriptDir = os.path.dirname(__file__)
    img = Image.open(os.path.join(dir_test_photos,''.join([str(test_photos_list[x]),'.jpg'])))
    
      
    # Get dimensions of the orignial image
    width, height = img.size  
    
    if width>=100 and height >=100:
        
        if width<height:
            # resize the image to width = number_of_pixels 
            wpercent = (number_of_pixels / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            rimg = img.resize((number_of_pixels, hsize), PIL.Image.ANTIALIAS)
            
            
        else:
            # resize the image to length = number_of_pixels 
            hpercent = (number_of_pixels / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            rimg = img.resize((wsize, number_of_pixels), PIL.Image.ANTIALIAS)
                
        # Get dimensions of the reduced image
        rwidth, rheight = rimg.size   
                
                # Get cropping parameters
        left = (rwidth - new_width)/2
        top = (rheight - new_height)/2
        right = (rwidth + new_width)/2
        bottom = (rheight + new_height)/2
                
        #crop image
        rcimg=rimg.crop((left, top, right, bottom))
                
        #show reduced and cropped image
       # plt.imshow(rcimg)
                
       #save the resulting image
        imagename1=''.join([str(test_photos_list[x]),'r.jpg'])
        imagename2=''.join([str(dir_test_photos_reduced),imagename1])
        rcimg.save(imagename2)
                
    else:
       imagename1toosmall=''.join([str(test_photos_list[x]),'r.jpg'])
       imagename2toosmall=''.join([str(dir_test_photos_reduced_too_small),imagename1toosmall])
       img.save(imagename2toosmall)
             