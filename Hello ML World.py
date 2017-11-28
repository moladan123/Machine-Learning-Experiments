import tensorflow as tf
import sklearn

a = tf.placeholder(tf.int64)
b = tf.placeholder(tf.int64)

adder = a + b * 3 + a ** 2

sess = tf.Session()

print(sess.run(adder, {a: 32, b: 2}))
print(sess.run(adder, {a: 3, b: 4.5}))
