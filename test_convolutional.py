import tensorflow as tf
import matplotlib.pyplot as pl
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def main(argv):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, reshape=False)
	###############################################################
	batch_size = 1000
	learning_rate = 0.01
	learning_rate_decay_rate = 0.95
	momentum = 0.9
	learning_epochs = 5
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
	W_max_initial = 0.1
	full1_dropout = 0.4
	full_regularization_param = 5e-4
	###############################################################
	#first convolutional layer
	conv1_W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, image_channels, conv1_filter_count], stddev=W_max_initial))
	conv1_b = tf.Variable(tf.zeros(shape=[conv1_filter_count]))
	#second convolutional layer
	conv2_W = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, conv1_filter_count, conv2_filter_count], stddev=W_max_initial))
	conv2_b = tf.Variable(tf.constant(conv2_b_initial, dtype=tf.float32, shape=[conv2_filter_count]))
	#first fully connected layer
	pooled_region_size = pool_size * pool_size
	full1_input_size = (image_width // pooled_region_size) * (image_height // pooled_region_size) * conv2_filter_count
	full1_W = tf.Variable(tf.truncated_normal(shape=[full1_input_size, full1_size], dtype=tf.float32, stddev=W_max_initial))
	full1_b	= tf.Variable(tf.constant(full_b_initial, dtype=tf.float32, shape=[full1_size]))
	#second fully connected layer (output)
	full2_W = tf.Variable(tf.truncated_normal(shape=[full1_size, full2_size], dtype=tf.float32, stddev=W_max_initial))
	full2_b = tf.Variable(tf.constant(full_b_initial, dtype=tf.float32, shape=[full2_size]))
	###############################################################
	#input data
	train_intput = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channels], name='train_intput')
	train_lables = tf.placeholder(tf.int32, shape=[batch_size,], name='train_lables')
	eval_input = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channels], name='eval_input')
	eval_labels = tf.placeholder(tf.int32, shape=[batch_size,], name='eval_labels')
	test_input = tf.placeholder(tf.float32, shape=[len(mnist.test.images), image_height, image_width, image_channels], name='test_input')
	test_labels = tf.placeholder(tf.int32, shape=[len(mnist.test.images),], name='test_labels')
	validation_input = tf.placeholder(tf.float32, shape=[len(mnist.validation.images), image_height, image_width, image_channels], name='validation_input')
	validation_labels = tf.placeholder(tf.int32, shape=[len(mnist.validation.images),], name='validation_labels')
	user_input = tf.placeholder(tf.float32, shape=[1, image_height, image_width, image_channels], name='user_input')
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
	batch_index = tf.Variable(0, dtype=tf.float32)
	learning_rate_decay = tf.train.exponential_decay(
		learning_rate, 
		batch_index, 
		len(mnist.train.images),          
	  	learning_rate_decay_rate, 
	  	staircase=True)
	#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	optimizer = tf.train.MomentumOptimizer(learning_rate_decay, momentum).minimize(cross_entropy)
  	#predictions for validation inputs
  	eval_prediction = tf.nn.softmax(model(eval_input, False))
  	#predictions for test inputs
  	test_prediction = tf.nn.softmax(model(test_input, False))
  	#predictions for validation inputs
  	validation_prediction = tf.nn.softmax(model(validation_input, False))
  	#predictions for single user input
  	user_prediction = tf.nn.softmax(model(user_input, False))
	###############################################################

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	batch_count = len(mnist.train.images) / batch_size
	#batch accuracy computation
	batch_prediction = tf.equal(tf.cast(eval_labels, tf.float32), tf.cast(tf.argmax(eval_prediction, 1), tf.float32))
	batch_accuracy = tf.reduce_mean(tf.cast(batch_prediction, tf.float32))
	#epoch accuracy computation (by validation set)
	epoch_prediction = tf.equal(tf.cast(validation_labels, tf.float32), tf.cast(tf.argmax(validation_prediction, 1), tf.float32))
	epoch_accuracy = tf.reduce_mean(tf.cast(epoch_prediction, tf.float32))
	#get user predicted digit value
	user_predicted_digit = tf.argmax(user_prediction, 1)
	#prepare arrays to save accuracy
	batch_accuracy_results = np.zeros([batch_count * learning_epochs])
	epoch_accuracy_results = np.zeros([learning_epochs])

	for e in range(0, learning_epochs):
		for i in range(0, batch_count):
			print('Run batch: ' + str(i) + ' in epoch: ' + str(e))
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			#get batch offset
			bi = i + e * batch_count
			#run SGD optimization for the current batch
			sess.run(optimizer, {train_intput: batch_x, train_lables: batch_y, batch_index: bi})
			#evaluate model on the same batch
			batch_accuracy_results[bi] = sess.run(batch_accuracy, {eval_input: batch_x, eval_labels: batch_y})
			print('Accuracy (batch: ' + str(i) + ' in epoch: ' + str(e) + '): '
									  + str(batch_accuracy_results[bi]))
		epoch_accuracy_results[e] = sess.run(epoch_accuracy, {validation_input: mnist.validation.images, validation_labels: mnist.validation.labels})
		print('Accuracy (in epoch: ' + str(e) + '): '
									  + str(epoch_accuracy_results[e]))

	#compute total accuracy onver test set
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
		test_feed = np.empty([1, image_height, image_width, 1])
		test_feed[0] = mnist.test.images[val_index]
		print('Predicted digit: ' + str(sess.run(user_predicted_digit, {user_input: test_feed})))
		print('Actual digit: ' + str(mnist.test.labels[val_index]))
		pl.ion()
		pl.imshow(mnist.test.images[val_index].reshape(image_height, image_width), cmap='gray')
		pl.show()
		pl.pause(0.001)
		pass

	pl.close()
	sess.close()

if __name__ == "__main__":
    main(sys.argv)