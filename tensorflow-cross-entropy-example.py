import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1] # S
one_hot_data = [1.0, 0.0, 0.0] # L

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# cross_entropy = - SUM( Li * log(Si))

cross_entropy = - tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as session:
	print(session.run(cross_entropy, feed_dict = { softmax: softmax_data, one_hot: one_hot_data }))
