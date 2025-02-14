#!/usr/bin/env python
# coding: utf-8

# In[5]:


from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Concatenate, Input
from keras.models import Model

from utils.attention_n_SN import SpectralNormalization, SelfAttention


# In[6]:


# 🛠 Encoder Block untuk Discriminator
def disc_block(x, filters, apply_batchnorm=True):
	x = SpectralNormalization(Conv2D(filters, kernel_size=4, strides=2, padding="same", use_bias=False))(x)
	if apply_batchnorm:
		x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	return x

# 🛠 PatchGAN Discriminator
def build_discriminator(input_shape=(256, 256, 1)):
	img_input = Input(shape=input_shape)
	cond_input = Input(shape=input_shape)  # Input kondisi (gambar asli)

	x = Concatenate()([img_input, cond_input])  # Menggabungkan input

	# Downsampling
	d1 = disc_block(x, 64, apply_batchnorm=False)  # Tanpa batchnorm untuk stabilitas awal
	d2 = disc_block(d1, 128)
	d3 = disc_block(d2, 256)
	d4 = disc_block(d3, 512)

	# Bottleneck dengan Self-Attention
	b = SelfAttention(512)(d4)

	# Final layer (1 output patch)
	out = SpectralNormalization(Conv2D(1, kernel_size=4, strides=1, padding="same", activation="sigmoid"))(b)

	return Model([img_input, cond_input], out, name="PatchGAN_Discriminator")


# 🛠 Build Model
discriminator = build_discriminator()
discriminator.summary()