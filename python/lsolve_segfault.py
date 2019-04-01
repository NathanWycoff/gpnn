
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

N = 10
A = tf.Variable(np.random.normal(size=[N,N]))
b = tf.Variable(np.random.normal(size=[N]))
tf.linalg.solve(A, b)

