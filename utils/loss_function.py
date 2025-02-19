#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

# In[2]:


alpha = 100

# L1 Loss
def l1_loss(y_true, y_pred):
	return tf.reduce_mean(tf.abs(y_true - y_pred))

# Generator Loss
def generator_loss(y_true, y_pred, fake_output):
	adv_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)
	l1 = l1_loss(y_true, y_pred)

	total_loss = alpha * l1 + adv_loss
	return total_loss

# Discriminator Loss
def discriminator_loss(real_output, fake_output):
	real_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
	fake_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)

	total_loss = real_loss + fake_loss
	return total_loss