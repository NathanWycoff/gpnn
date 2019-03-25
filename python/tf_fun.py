

num_points = 100
# Index points should be a collection (100, here) of feature vectors. In this
# example, we're using 1-d vectors, so we just need to reshape the output from
# np.linspace, to give a shape of (100, 1).
index_points = np.expand_dims(np.linspace(-1., 1., num_points), -1)

# Define a kernel with default parameters.
kernel = psd_kernels.ExponentiatedQuadratic()

gp = tfd.GaussianProcess(kernel, index_points)

samples = gp.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`

noisy_gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=index_points,
    observation_noise_variance=.05)
noisy_samples = noisy_gp.sample(10)
# ==> 10 independently drawn, noisy joint samples at `index_points`

# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1).astype(np.float32)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.get_variable('amplitude', 1, dtype = np.float32),
    length_scale=tf.get_variable('length_scale', 1, dtype = np.float32))

gp = tfd.GaussianProcess(kernel, observed_index_points)
neg_log_likelihood = -gp.log_prob(observed_values)

optimize = tf.train.AdamOptimizer().minimize(neg_log_likelihood)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(1000):
    _, neg_log_likelihood_ = sess.run([optimize, neg_log_likelihood])
    if i % 100 == 0:
      print("Step {}: NLL = {}".format(i, neg_log_likelihood_))
  print("Final NLL = {}".format(neg_log_likelihood_))

