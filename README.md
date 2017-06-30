# generating images using Auto-Encoder
## Description ##
This project is about how to generate MNIST images using Auto Encoder from MNIST Image.
Image generation can be useful for reconstructing poor quality image or improve existing image.
MNIST File can be found at [Yann Lecun](http://yann.lecun.com/exdb/mnist/).
This project is developed by using [Tensorflow](http://tensorflow.org).

## Methods ##

## How Program Works ##
- import libraries: *tensorflow, numpy and matplotlib*.
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
...
```
- set *variables* as follow
```
num_epoch = 1000 #num of epoch
img_w = 28 #image width
img_h = 28 #image height
hidden_1 = 256 #hidden layer 1
...
input_size=img_w * img_h
```
- load *MNIST dataset*
```
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
```
- define *weight and biases* in python dictionary. you can give random initialization stddev=0.01.
```
weights = { 'encoder1': tf.Variable(tf.truncated_normal([input_size,hidden_1],stddev=0.01)), #connect 28 * 28 --> 256
		'encoder2': tf.Variable(tf.truncated_normal([hidden_1,hidden_2],stddev=0.01)), #connect 256 --> 128
		... }
biases = {'encoder1': tf.Variable(tf.truncated_normal([hidden_1],stddev=0.01)),
		'encoder2': tf.Variable(tf.truncated_normal([hidden_2],stddev=0.01)),
    		... }
```
- connect *weight and biases * with neural network operation x * w + b. you can make it in layers as follow:
```
l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder1']),biases['encoder1'])) #layer 1 = x * w1 + b
l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['encoder2']),biases['encoder2']))#layer 2 = l1 * w2 + b
	...
l6 = tf.nn.relu(tf.add(tf.matmul(l5, weights['decoder3']),biases['decoder3']))#layer 6 = l5 * w6 + b
	
```
- set y_predict = generating image and y_true = input data.
```
y_predict = model(x) #fill y_predict with model
y_true = x #label = x (input)
```
- set cost function (MSE) and optimizer (RMSPropOptimizer). you also can use another type of cost function and optimizer like [Adam](https://www.tensorflow.org/api_docs/p…) or [AdaGrad](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
```
cost = tf.reduce_mean(tf.pow(tf.subtract(y_true,y_predict),2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
	
```
- training model using *Session* as follow:
```
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
```
index = np.random.randint(mnist.test.images.shape[0], size=8) #random 8 number from test data to be predicted
autoencoder = sess.run(y_predict, feed_dict={x: mnist.test.images[index]}) #predicted test data using model
```		
- Save reconstructed images
```
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
