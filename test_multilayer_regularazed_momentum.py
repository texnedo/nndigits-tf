import tensorflow as tf
import matplotlib.pyplot as pl
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main(argv):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	###############################################################
	l_size = [784, 100, 10]
	batch_size = 100
	learning_rate = 0.01
	learning_epochs = 15
	regularization_rate = 5
	dropout_keep_probability = 0.9
	momentum = 0.9
	###############################################################
	print ("Network shape: " + str(l_size))
	l_count = len(l_size)
	a = [None] * (l_count - 1)
	W = [None] * (l_count - 1)
	b = [None] * (l_count - 1)

	for l in range(0, l_count - 1):
		a[l] = tf.placeholder(tf.float32, [None, l_size[l]])
	a0 = a[0]
	y = tf.placeholder(tf.float32, [None, l_size[l_count - 1]])

	keep_prob = tf.placeholder(tf.float32)
	for l in range(0, l_count - 1):
		W[l] = tf.Variable(tf.random_normal([l_size[l], l_size[l + 1]]))
		if l != 0:
			W[l] = tf.nn.dropout(W[l], keep_prob)
		b[l] = tf.Variable(tf.random_normal([l_size[l + 1]]))

	for l in range(1, l_count - 1):
		a[l] = tf.nn.relu(tf.add(tf.matmul(a[l - 1], W[l - 1]), b[l - 1]))

	y_pred = tf.add(tf.matmul(a[l_count - 2], W[l_count - 2]), b[l_count - 2])

	reg = tf.zeros([1], tf.float32)
	for l in range(0, l_count - 1):
		reg = reg + tf.nn.l2_loss(W[l])
	reg = reg * regularization_rate / (len(mnist.train.images))
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y) + reg)
	result = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cross_entropy)
	###############################################################

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	batch_count = len(mnist.train.images) / batch_size
	batch_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
	batch_accuracy = tf.reduce_mean(tf.cast(batch_prediction, tf.float32))
	
	batch_accuracy_results = np.zeros([batch_count * learning_epochs])

	for e in range(0, learning_epochs):
		for i in range(0, batch_count):
			print('Run batch: ' + str(i) + ' in epoch: ' + str(e))
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			sess.run(result, {a0 : batch_x, y : batch_y, keep_prob : dropout_keep_probability})
			bi = i + e * batch_count
			batch_accuracy_results[bi] = sess.run(batch_accuracy, {a0: batch_x, y: batch_y, keep_prob : 1.0})
			print('Accuracy (batch: ' + str(i) + ' in epoch: ' + str(e) + '): '
									  + str(batch_accuracy_results[bi]))
	
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print('Accuracy: ' + str(sess.run(accuracy, {a0: mnist.test.images, y: mnist.test.labels, keep_prob : 1.0})))

	pl.ion()
	pl.plot(batch_accuracy_results)
	pl.show()
	pl.pause(0.001)

	print('Press any key to show results.')
	sys.stdin.readline()
	pl.close()

	print('Press enter to advance through test images (and any other key to exit).')
	while sys.stdin.readline() == '\n':
		val_index = tf.random_uniform(shape=[1], maxval=len(mnist.test.images)-1, dtype=tf.int32);
		val_index = int(val_index.eval())
		print('Random image index: ' + str(val_index))

		test_feed = mnist.test.images[val_index].reshape(1, 784)

		print(mnist.test.images[val_index].shape)
		print(test_feed.shape)

		test_eval = tf.argmax(y_pred, 1)
		print('Predicted digit: ' + str(sess.run(test_eval, {a0: test_feed, keep_prob : 1.0})))

		img1 = test_feed.reshape(28, 28)
		pl.ion()
		pl.imshow(img1, cmap='gray')
		pl.show()
		pl.pause(0.001)
		pass

	pl.close()
	sess.close()


if __name__ == "__main__":
    main(sys.argv)