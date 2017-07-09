import tensorflow as tf
import matplotlib.pyplot as pl
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main(argv):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	#idx = int(sys.stdin.readline())
	#for pos in range(0, len(mnist.train.labels[idx]) - 1):
	#	if mnist.train.labels[idx][pos] == 1:
	#		sys.stdout.write('Digit: ' + str(pos) + '\n')
	#		break
	#img1 = mnist.train.images[idx].reshape(28, 28)
	#pl.imshow(img1, cmap='gray')
	#pl.show()
	#sys.stdin.readline()
	#pl.close()
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	W = tf.Variable(tf.zeros([784, 10], tf.float32))
	b = tf.Variable(tf.zeros([10], tf.float32))
	y_pred = tf.matmul(x, W) + b
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
	result = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	for i in range(0, 1000):
		print 'run batch: ' + str(i)
		batch_x, batch_y = mnist.train.next_batch(100)
		sess.run(result, {x : batch_x, y : batch_y})
	
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print('Accuracy: ' + str(sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})))

	while sys.stdin.readline() == '\n':
		val_index = tf.random_uniform(shape=[1], maxval=len(mnist.test.images)-1, dtype=tf.int32);
		val_index = int(val_index.eval())
		print('Random x index: ' + str(val_index))

		test_feed = mnist.test.images[val_index].reshape(1, 784)

		print(mnist.test.images[val_index].shape)
		print(test_feed.shape)

		test_eval = tf.argmax(y_pred, 1)
		print('Eval: ' + str(sess.run(test_eval, {x: test_feed})))

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