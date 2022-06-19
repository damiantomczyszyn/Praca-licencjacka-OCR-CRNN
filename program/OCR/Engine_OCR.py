#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from SegmentPage import segment_into_lines
from SegmentLine import segment_into_words
from RecognizeWord import recognize_words
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")#,default='all')
args = vars(ap.parse_args())
# In[2]:


#line_img_array=segment_into_lines('../../test_data/test_image2.jpeg')
line_img_array=segment_into_lines(args["image"])

full_index_indicator=[]
all_words_list=[]

len_line_arr=0


# In[3]:


for idx,im in enumerate(line_img_array):
    line_indicator,word_array=segment_into_words(im,idx)
    for k in range(len(word_array)):
        full_index_indicator.append(line_indicator[k])
        all_words_list.append(word_array[k])   
        
    len_line_arr+=1
    

all_words_list=np.array(all_words_list)



# In[4]:
recognize_words(full_index_indicator,all_words_list/255.,len_line_arr)

