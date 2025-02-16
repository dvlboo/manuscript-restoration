#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


# metrik psnr
def psnr(y_true, y_pred):
	return tf.image.psnr(y_true, y_pred, max_val=1.0)

# metrik ssim
def ssim(y_true, y_pred):
	return tf.image.ssim(y_true, y_pred, max_val=1.0)
