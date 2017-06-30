# Image Generation using Auto-Encoder
## Description ##
This project is about how to generate MNIST images using Auto Encoder from MNIST Image.
Image generation can be useful for reconstructing poor quality image or improve existing image.
MNIST File can be found at [Yann Lecun](http://yann.lecun.com/exdb/mnist/).
This project is developed by using [Python3.6](https://www.python.org/downloads/release/python-360/), [Tensorflow](http://tensorflow.org) and Matplotlib 

## Methods ##
An Auto-Encoder Neural Network is Unsupervised Learning that can be used to reconstruct images. suppose there is no label for each data x = {x1,x2,..,xn} and we want to reconstruct x = x. Auto Encoder architecture can be seen as follow:
![Fig.1](https://raw.github.com/tavgreen/generating_images/master/architecture.png?raw=true "Auto Encoder")

above architecture describes MNIST image with size 28 * 28 will be reconstructed into MNIST image. each input pixel (28 * 28 = 784 pixel) will be forwared into hidden 1 until hidden 3. Hidden 1 consists of 256 neurons, hidden 2 consists 128 neurons and so on. Each neuron (blue circle) in hidden consists of weights and biases from previous layer (blue line). Neuron can be calculated as follows:

![Fig.2](https://raw.github.com/tavgreen/generating_images/master/formula.png?raw=true "Auto Encoder")

in above picture, let say pixel[0,0]=1 and pixel[0,1]=0 from MNIST image. each pixels (784 pixels) will be mapped into one neuron(x1 or x2 or .. xn).  each x has weight and bias that will be connected into next layer let say h1. h1 can be calculated as above picture. after that, sigmoid function as activation function will be worked as above picture.

after mapping into output layer (reconstructed image), loss function will be calculated using Mean Square Error (MSE) in order to calculate backpropagation and update weights and biases in previous layer. The calculation of loss function and calling backpropagation will be repeated until the end of epoch or until convergence.

## How Program Works ##
- import libraries: *tensorflow, numpy and matplotlib*.
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
...
```
- set *variables* as follow
```python
num_epoch = 1000 #num of epoch
img_w = 28 #image width
img_h = 28 #image height
hidden_1 = 256 #hidden layer 1
...
input_size=img_w * img_h
```
- load *MNIST dataset*
```python
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
```
- define *weight and biases* in python dictionary. you can give random initialization stddev=0.01.
```python
weights = { 'encoder1': tf.Variable(tf.truncated_normal([input_size,hidden_1],stddev=0.01)), #connect 28 * 28 --> 256
		'encoder2': tf.Variable(tf.truncated_normal([hidden_1,hidden_2],stddev=0.01)), #connect 256 --> 128
		... }
biases = {'encoder1': tf.Variable(tf.truncated_normal([hidden_1],stddev=0.01)),
		'encoder2': tf.Variable(tf.truncated_normal([hidden_2],stddev=0.01)),
    		... }
```
- connect *weight and biases * with neural network operation x * w + b. you can make it in layers as follow:
```python
l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder1']),biases['encoder1'])) #layer 1 = x * w1 + b
l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['encoder2']),biases['encoder2']))#layer 2 = l1 * w2 + b
	...
l6 = tf.nn.relu(tf.add(tf.matmul(l5, weights['decoder3']),biases['decoder3']))#layer 6 = l5 * w6 + b
	
```
- set y_predict = generating image and y_true = input data.
```python
y_predict = model(x) #fill y_predict with model
y_true = x #label = x (input)
```
- set cost function (MSE) and optimizer (RMSPropOptimizer). you also can use another type of cost function and optimizer like [Adam](https://www.tensorflow.org/api_docs/p…) or [AdaGrad](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
```python
cost = tf.reduce_mean(tf.pow(tf.subtract(y_true,y_predict),2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)	
```
- training model using *Session* as follow:
```python
with tf.Session() as sess: #running session
	sess.run(tf.global_variables_initializer()) #run all variable
	for epoch in range(num_epoch): #loop 1000x
		epoch_loss = 0 #variable to store epoch_loss
		for _ in range(int(mnist.train.num_examples / batch_size)):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x}) #feed forward
			epoch_loss += c #count loss
	correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true,1)) #if y_predict equals y_true than correct
```
- running test data using training model above as follows:
```python
index = np.random.randint(mnist.test.images.shape[0], size=8) #random 8 number from test data to be predicted
autoencoder = sess.run(y_predict, feed_dict={x: mnist.test.images[index]}) #predicted test data using model
```		
- Save reconstructed images
```python
for i in range(8):
	plt.imsave("result"+str(i)+".png",np.reshape(autoencoder[i],(img_h,img_h)), cmap=plt.get_cmap('gray'))
```
## Results ##

## Future Works ##
you can generate or reconstruct images using your own dataset. the example can be found list below:
-  Generate Pokemon images by [@musicmilif](https://github.com/musicmilif/Pokemon-Generator)
-  Generate MNIST images using Variational-Auto Encoder by [@kvfrans](https://github.com/kvfrans/variational-autoencoder)

the paper about image generation can be found here:
- DRAW Recurrent Neural Network [Gregor et al](https://arxiv.org/abs/1502.04623)
- Tutorial Variational Auto Encoder [Doersch](https://arxiv.org/abs/1606.05908)
