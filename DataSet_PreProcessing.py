
# coding: utf-8

# # Preprocessing of Data with resizing 

# In[1]:


from os import listdir
from os.path import isfile, join
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt 
import tensorflow as tf
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import PIL
from PIL import Image


# In[2]:


count = 0;
numberOfSlices = 30;
numberOfImages = 274;
j = 0;
output_array = np.zeros((numberOfImages*numberOfSlices,1), dtype=np.float32)

def getFiles(path, f):
    if isfile(join(path,f)) :
        if(f.endswith('.mha') and 'OT.' in f):
            global count,numberOfSlices,j,output_array;
            count = count + 1;
            itkimage = sitk.ReadImage(join(path,f));
            dataArray = sitk.GetArrayFromImage(itkimage);
            i = 0;
            for i in range(numberOfSlices):
                if((dataArray[80+i,:,:] == 0).all()):
                    output_array[j*numberOfSlices+ i] =0;
                else:
                    output_array[j*numberOfSlices+ i] =1;
            j = j + 1;
    else:
        for file in listdir(join(path,f)):
            getFiles(join(path,f),file);

            
getFiles("/home/ee/mtech/eet182559/","BRATS2015_Training");
print(count);


# In[3]:


count = 0;
numberOfSlices = 30;
numberOfImages = 274;
j = 0;
input_array = np.zeros((numberOfImages * numberOfSlices, 240, 240), dtype = np.float32)

def getFiles(path, f):
    if isfile(join(path,f)) :
        if(f.endswith('.mha') and 'VSD.Brain.XX.O.MR_T1.' in f):
            global count,numberOfSlices,j,output_array;
            count = count + 1;
            itkimage = sitk.ReadImage(join(path,f));
            dataArray = sitk.GetArrayFromImage(itkimage);
            i = 0;
            for i in range(numberOfSlices):
                s = f.replace(".mha","_") + str(i) + ".png"
                plt.imsave(s,dataArray[80+i,:,:],cmap = 'gray')
            j = j + 1;
    else:
        for file in listdir(join(path,f)):
            getFiles(join(path,f),file);

            
getFiles("/home/ee/mtech/eet182559/","BRATS2015_Training");
print(count);

