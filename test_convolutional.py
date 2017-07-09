import tensorflow as tf
import matplotlib.pyplot as pl
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main(argv):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape=False)
	###############################################################
	batch_size = 50
	max_batch_per_epoch = 50
	#batch_size = 1
	learning_rate = 0.01
	learning_epochs = 10
	#learning_epochs = 1
	momentum = 0.9
	image_width = 28
	image_height = 28
	image_channels = 1
	image_labels_count = 10
	###############################################################
	filter_size = 5
	pool_size = 2
	conv1_filter_count = 32
	conv2_filter_count = 64
	conv2_b_initial = 0.1
	full1_size = 512
	full2_size = image_labels_count
	full_b_initial = 0.1
	full1_dropout = 0.5
	full_regularization_param = 5e-4
	###############################################################
	#first convolutional layer
	conv1_W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, image_channels, conv1_filter_count]))
	conv1_b = tf.Variable(tf.zeros(shape=[conv1_filter_count]))
	#second convolutional layer
	conv2_W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, conv1_filter_count, conv2_filter_count]))
	conv2_b = tf.Variable(tf.constant(conv2_b_initial, dtype=tf.float32, shape=[conv2_filter_count]))
	#first fully connected layer
	pooled_region_size = pool_size * pool_size
	full1_input_size = (image_width // pooled_region_size) * (image_height // pooled_region_size) * conv2_filter_count
	full1_W = tf.Variable(tf.truncated_normal(shape=[full1_input_size, full1_size], dtype=tf.float32))
	full1_b	= tf.Variable(tf.constant(full_b_initial, dtype=tf.float32, shape=[full1_size]))
	#second fully connected layer (output)
	full2_W = tf.Variable(tf.truncated_normal(shape=[full1_size, full2_size], dtype=tf.float32))
	full2_b = tf.Variable(tf.constant(full_b_initial, dtype=tf.float32, shape=[full2_size]))
	###############################################################
	#input data
	train_intput = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channels], name='train_intput')
	train_lables = tf.placeholder(tf.int32, shape=[batch_size,], name='train_lables')
	eval_input = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channels], name='eval_input')
	eval_labels = tf.placeholder(tf.int32, shape=[batch_size,], name='eval_labels')
	test_input = tf.placeholder(tf.float32, shape=[len(mnist.test.images), image_height, image_width, image_channels], name='test_input')
	test_labels = tf.placeholder(tf.int32, shape=[len(mnist.test.images),], name='test_labels')
	###############################################################
	def model(data, train=False):
		#first convolutional layer
		result = tf.nn.conv2d(data, conv1_W, [1, 1, 1, 1], 'SAME')
		result = tf.nn.relu(tf.nn.bias_add(result, conv1_b))
		result = tf.nn.max_pool(result, [1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], 'SAME')
		#second convolutional layer
		result = tf.nn.conv2d(result, conv2_W, [1, 1, 1, 1], 'SAME')
		result = tf.nn.relu(tf.nn.bias_add(result, conv2_b))
		result = tf.nn.max_pool(result, [1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], 'SAME')
		#first fully connected layer
		shape = result.get_shape().as_list()
		#feed data into a single vector
		result = tf.reshape(result, [shape[0], shape[1] * shape[2] * shape[3]])
		result = tf.nn.relu(tf.add(tf.matmul(result, full1_W), full1_b))
	 	if train:
			result = tf.nn.dropout(result, full1_dropout)
	    #second fully connected layer
		result = tf.add(tf.matmul(result, full2_W), full2_b)
		return result
	###############################################################
	#feed test inputs
	out = model(train_intput, True)
	#design loss (cross entropy) function for the net
	cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, train_lables))
	#regularization for weights
  	regularizers = tf.nn.l2_loss(full1_W) + tf.nn.l2_loss(full1_b) + tf.nn.l2_loss(full2_W) + tf.nn.l2_loss(full2_b)
  	#add the regularization term to the loss.
  	cross_entropy += full_regularization_param * regularizers
  	#run gradient descent
  	optimizer = tf.train.AdamOptimizer(learning_rate, momentum).minimize(cross_entropy)
  	#predictions for validation inputs
  	eval_prediction = tf.nn.softmax(model(eval_input, False))
  	#predictions for test inputs
  	test_prediction = tf.nn.softmax(model(test_input, False))
	###############################################################

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	batch_count = len(mnist.train.images) / batch_size
	batch_count = min(batch_count, max_batch_per_epoch)
	batch_prediction = tf.equal(tf.cast(eval_labels, tf.float32), tf.cast(tf.argmax(eval_prediction, 1), tf.float32))
	batch_accuracy = tf.reduce_mean(tf.cast(batch_prediction, tf.float32))
	
	batch_accuracy_results = np.zeros([batch_count * learning_epochs])

	for e in range(0, learning_epochs):
		for i in range(0, batch_count):
			print('Run batch: ' + str(i) + ' in epoch: ' + str(e))
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			#run SGD optimization for the current batch
			sess.run(optimizer, {train_intput : batch_x, train_lables : batch_y})
			#get batch offset
			bi = i + e * batch_count
			#evaluate model on the same batch
			batch_accuracy_results[bi] = sess.run(batch_accuracy, {eval_input: batch_x, eval_labels: batch_y})
			print('Accuracy (batch: ' + str(i) + ' in epoch: ' + str(e) + '): '
									  + str(batch_accuracy_results[bi]))

	total_prediction = tf.equal(tf.cast(test_labels, tf.float32), tf.cast(tf.argmax(test_prediction, 1), tf.float32))
	total_accuracy = tf.reduce_mean(tf.cast(total_prediction, tf.float32))
	print('Total accuracy: ' + str(sess.run(total_accuracy, {test_input: mnist.test.images, test_labels: mnist.test.labels})))

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

		test_eval = tf.argmax(eval_prediction, 1)
		print('Predicted digit: ' + str(sess.run(test_eval, {eval_input: test_feed})))

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