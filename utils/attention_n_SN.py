#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from keras.layers import Conv2D


# In[2]:


# 🌟 Spectral Normalization Layer
class SpectralNormalization(tf.keras.layers.Wrapper):
	def __init__(self, layer, power_iterations=1, **kwargs):
		super(SpectralNormalization, self).__init__(layer, **kwargs)
		self.power_iterations = power_iterations

	def build(self, input_shape):
		self.layer.build(input_shape)
		self.kernel = self.layer.kernel
		self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
															initializer="random_normal",
															trainable=False,
															name="sn_u")

	def call(self, inputs, training=None):
		# Power Iteration for Spectral Normalization
		w_shape = self.kernel.shape.as_list()
		w = tf.reshape(self.kernel, [-1, w_shape[-1]])

		u = self.u
		for _ in range(self.power_iterations):
			v = tf.linalg.l2_normalize(tf.matmul(u, tf.transpose(w)))
			u = tf.linalg.l2_normalize(tf.matmul(v, w))

		sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
		w_norm = w / sigma
		self.layer.kernel.assign(tf.reshape(w_norm, w_shape))
		self.u.assign(u)
		return self.layer(inputs, training=training)


# 🔥 Self-Attention Layer
class SelfAttention(tf.keras.layers.Layer):
	def __init__(self, filters):
		super(SelfAttention, self).__init__()
		self.filters = filters
		self.query = Conv2D(filters // 8, kernel_size=1, padding="same")
		self.key = Conv2D(filters // 8, kernel_size=1, padding="same")
		self.value = Conv2D(filters, kernel_size=1, padding="same")
		self.gamma = tf.Variable(initial_value=tf.zeros(1), trainable=True)

	def call(self, x):
		batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
		q = tf.reshape(self.query(x), (batch, height * width, channels // 8))
		k = tf.reshape(self.key(x), (batch, height * width, channels // 8))
		v = tf.reshape(self.value(x), (batch, height * width, channels))

		attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
		attn_out = tf.matmul(attn, v)
		attn_out = tf.reshape(attn_out, (batch, height, width, channels))

		return self.gamma * attn_out + x

