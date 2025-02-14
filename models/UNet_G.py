#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)

from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, ReLU, Input, Dropout, Concatenate
from keras.models import Model

from utils.attention_n_SN import SelfAttention, SpectralNormalization

# In[2]:


# 🌟 Encoder Block
def encoder_block(x, filters, apply_batchnorm=True):
	x = SpectralNormalization(Conv2D(filters, kernel_size=4, strides=2, padding="same", use_bias=False))(x)
	if apply_batchnorm:
		x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	return x


# 🌟 Decoder Block
def decoder_block(x, skip_input, filters, apply_dropout=False):
	x = SpectralNormalization(Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", use_bias=False))(x)
	x = BatchNormalization()(x)
	if apply_dropout:
		x = Dropout(0.5)(x)
	x = ReLU()(x)
	x = Concatenate()([x, skip_input])
	return x


# 🌟 U-Net Generator Model
def build_generator(input_shape=(256, 256, 1)):
	inputs = Input(shape=input_shape)

	# Encoder (Downsampling)
	e1 = encoder_block(inputs, 64, apply_batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	e6 = encoder_block(e5, 512)
	e7 = encoder_block(e6, 512)

	# Bottleneck with Self-Attention
	b = encoder_block(e7, 512)
	b = SelfAttention(512)(b)

	# Decoder (Upsampling + Skip Connections)
	d1 = decoder_block(b, e7, 512, apply_dropout=True)
	d2 = decoder_block(d1, e6, 512, apply_dropout=True)
	d3 = decoder_block(d2, e5, 512, apply_dropout=True)
	d4 = decoder_block(d3, e4, 512)
	d5 = decoder_block(d4, e3, 256)
	d6 = decoder_block(d5, e2, 128)
	d7 = decoder_block(d6, e1, 64)

	outputs = SpectralNormalization(Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh"))(d7)

	return Model(inputs, outputs, name="U-Net_Generator")


# 🛠 Build Model
generator = build_generator()
generator.summary()

