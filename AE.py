'''
by Octaviano Pratama
reference: tensorflow.org
this program is for Autoencoder program testing for MNIST dataset
it can be used for image generation: input x output x
every training image in MNIST will be calculated pixel by pixel 28 * 28 using this algorithm
there are 3 hidden layer for encoder and 3 hidden layer for decoder (reconstructed)
'''
import tensorflow as tf #load tensorflow library, aliases as tf
from tensorflow.examples.tutorials.mnist import input_data #load mnist library
import numpy as np #load numpy library
import matplotlib.pyplot as plt #load matplotlib library to plot or save generated image
import matplotlib.image as mpimg #load matplotlib image library

num_epoch = 1000 #num of epoch
img_w = 28 #image width
img_h = 28 #image height
hidden_1 = 256 #hidden layer 1
hidden_2 = 128 #hidden layer 2
hidden_3 = 64 #hidden layer 3
learning_rate = 0.001 #learning rate
batch_size = 50 #batch size
input_size = img_w * img_h #input size 28 * 28 MNIST
mnist = input_data.read_data_sets("/tmp/data",one_hot=True) #load mnist data. one_hot: 0010000000 = 2, 0001000000 = 3 and so on
x = tf.placeholder('float', [None, input_size]) #input : give a placeholder with size 28 * 28
def model(x): #definition of model
	weights = { #definition weight #stddev to give random value to each weight.
		'encoder1': tf.Variable(tf.truncated_normal([input_size,hidden_1],stddev=0.01)), #connect 28 * 28 --> 256
		'encoder2': tf.Variable(tf.truncated_normal([hidden_1,hidden_2],stddev=0.01)), #connect 256 --> 128
		'encoder3': tf.Variable(tf.truncated_normal([hidden_2,hidden_3],stddev=0.01)), #connect 128 --> 64 
		'decoder1': tf.Variable(tf.truncated_normal([hidden_3,hidden_2],stddev=0.01)), #connect 64 --> 128 (reconstructed)
		'decoder2': tf.Variable(tf.truncated_normal([hidden_2,hidden_1],stddev=0.01)), #connect 127 --> 256
		'decoder3': tf.Variable(tf.truncated_normal([hidden_1,input_size],stddev=0.01)) #connect 256 --> 28 * 28
	}
	biases = { #definition of biases
		'encoder1': tf.Variable(tf.truncated_normal([hidden_1],stddev=0.01)),
		'encoder2': tf.Variable(tf.truncated_normal([hidden_2],stddev=0.01)),
		'encoder3': tf.Variable(tf.truncated_normal([hidden_3],stddev=0.01)),
		'decoder1': tf.Variable(tf.truncated_normal([hidden_2],stddev=0.01)),
		'decoder2': tf.Variable(tf.truncated_normal([hidden_1],stddev=0.01)),
		'decoder3': tf.Variable(tf.truncated_normal([input_size],stddev=0.01))
	}
	l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder1']),biases['encoder1'])) #layer 1 = x * w1 + b
	l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['encoder2']),biases['encoder2']))#layer 2 = l1 * w2 + b
	l3 = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['encoder3']),biases['encoder3']))#layer 3 = l2 * w3 + b
	l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, weights['decoder1']),biases['decoder1']))#layer 4 = l3 * w4 + b
	l5 = tf.nn.sigmoid(tf.add(tf.matmul(l4, weights['decoder2']),biases['decoder2']))#layer 5 = l4 * w5 + b
	l6 = tf.nn.relu(tf.add(tf.matmul(l5, weights['decoder3']),biases['decoder3']))#layer 6 = l5 * w6 + b
	return l6 #return l6 as final model
def training(x):
	y_predict = model(x) #fill y_predict with model
	y_true = x #label = x (input)
	cost = tf.reduce_mean(tf.pow(tf.subtract(y_true,y_predict),2)) #calculate cost function using MSE: (y_true - y_predict) square
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost) #calculate optimization using RMSPropOptimizer using learning rate, minimize cost
	with tf.Session() as sess: #running session: running the model
		sess.run(tf.global_variables_initializer()) #run initialize global variables: cost, optimizer
		for epoch in range(num_epoch): #loop epoch
			epoch_loss = 0 #loss epoch
			for _ in range(int(mnist.train.num_examples / batch_size)): #loop batch
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)  #epoch_x: data
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x}) #compute x, y using optimizer and cost save cost in c
				epoch_loss += c #sum loss
			print('Epoch ',epoch, ' ,Loss: ',epoch_loss) #print epoch and loss
		correct = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true,1)) # y' = y then correct
		accuracy = tf.reduce_mean(tf.cast(correct, 'float')) #cast to float correct
		print('Accuracy: ',accuracy.eval({x:mnist.test.images, x:mnist.test.images})) #eval using x
		#test data
		index = np.random.randint(mnist.test.images.shape[0], size=8) #random 8 number from test data to be predicted
		autoencoder = sess.run(y_predict, feed_dict={x: mnist.test.images[index]}) #predicted test data using model
		for i in range(8):
			plt.imsave("result"+str(i)+".png",np.reshape(autoencoder[i],(img_h,img_h)), cmap=plt.get_cmap('gray')) #save result
training(x)