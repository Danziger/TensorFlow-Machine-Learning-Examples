import tensorflow as tf

def calculate():
	output = None
	logits = tf.placeholder(tf.float32)
	values = [2.0, 1.0, 0.1]

	softmax = tf.nn.softmax(logits)  

	with tf.Session() as session:
		output = session.run(softmax, feed_dict = { logits: values })

	return output

print(calculate())
