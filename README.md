# generating images using Auto-Encoder
## Description ##
This project is about how to generate MNIST images using Auto Encoder from MNIST Image.
Image generation can be useful for reconstructing poor quality image or improve existing image.
MNIST File by [Yann Lecun] can be found at (http://yann.lecun.com/exdb/mnist/).
This project is developed by using Tensorflow (http://tensorflow.org).

## Methods ##

## How Program Works ##
- import libraries: tensorflow, numpy and matplotlib.
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
...
```
- set variables as follow
```
num_epoch = 1000 #num of epoch
img_w = 28 #image width
img_h = 28 #image height
hidden_1 = 256 #hidden layer 1
...
input_size=img_w * img_h
```
- load MNIST dataset
```
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
```
- define weight and biases in python dictionary.give random initialization stddev=0.01.
```
weights = { #definition weight #stddev to give random value to each weight.
		'encoder1': tf.Variable(tf.truncated_normal([input_size,hidden_1],stddev=0.01)), #connect 28 * 28 --> 256
		'encoder2': tf.Variable(tf.truncated_normal([hidden_1,hidden_2],stddev=0.01)), #connect 256 --> 128
		... }
biases = { #definition of biases
		'encoder1': tf.Variable(tf.truncated_normal([hidden_1],stddev=0.01)),
		'encoder2': tf.Variable(tf.truncated_normal([hidden_2],stddev=0.01)),
    ... }
```
- connect weight and biases and make it in layer as follow:
```
l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder1']),biases['encoder1'])) #layer 1 = x * w1 + b
l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['encoder2']),biases['encoder2']))#layer 2 = l1 * w2 + b
	...
l6 = tf.nn.relu(tf.add(tf.matmul(l5, weights['decoder3']),biases['decoder3']))#layer 6 = l5 * w6 + b
	
```
## Results ##

## Future Works ##
