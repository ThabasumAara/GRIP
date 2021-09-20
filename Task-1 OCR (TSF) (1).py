#!/usr/bin/env python
# coding: utf-8

# # #GRIP_SEPTEMBER 21
# 
# # NAME: THABASUM AARA S
# 
# # DOMAIN: COMPUTER VISION AND IOT
# 
# # TASK-1 OPTICAL CHARACTER RECOGNITION (OCR)
# 

# Import modules 

# In[1]:


import cv2 #an image preprocessing library


# In[2]:


import pytesseract #an image to text library 


# In[3]:


import numpy as np #used for mathematics but can be used in image processing 


# In[4]:


import matplotlib.pyplot as plt
import matplotlib.image as img #display an image


# In[5]:


# Configure the module
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[6]:


# Make the image grey
# opening an image from the source path
img = cv2.imread(r'E:\Task-1 Optical Character Recognition (OCR)/handwritten.png')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)
kernel = np.ones((2, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)


# In[7]:


# Use OCR to read the text from the image
out_below = pytesseract.image_to_string(img)
# Print the text
print(out_below)


# In[8]:



# write text in a text file and save it to source path   
with open('handwritten2text.txt', mode ='w') as file:
    file.write(out_below)
    print('Done')


# Printed Image to Text converter

# In[9]:


# Import some modules
import cv2 
# An image proccessing library
import pytesseract
# an image to text library
import numpy as np 
# used for mathematics but can be used in image proccessing
import matplotlib.pyplot as plt
import matplotlib.image as img

# Configure the module
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Make the image grey
img = cv2.imread(r'E:\Task-1 Optical Character Recognition (OCR)\test.png')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)
kernel = np.ones((2, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)

# converts the image to result and saves it into result variable
out_below = pytesseract.image_to_string(img)

# Print the text
print(out_below)

# write text in a text file and save it to source path   
with open('printed2text.txt', mode ='w') as file:
    file.write(out_below)
    print('Done')


# In[ ]:




