import tensorflow as tf

hello = tf.constant('Hello World!')

with tf.Session() as session:
    print(session.run(hello))
	
