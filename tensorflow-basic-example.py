import tensorflow as tf

def calculate():
	output = None
	a = tf.cast(tf.constant(10), tf.float64)
	b = tf.cast(tf.constant(2.0), tf.float64)
	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	
	# (a + x) / b  - a * y
	
	z = tf.subtract(
		tf.divide(tf.add(a, x), b),
		tf.multiply(a, y)
	)
	
	with tf.Session() as session:
		output = session.run(z, feed_dict = { x: 100, y: 1.5 })

	return output

print(calculate())
