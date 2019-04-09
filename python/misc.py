import tensorflow as tf

def random_nn(P, L, H, R, act = tf.nn.tanh):
    """
    Create a neural net with default random weights.

    P dimension of inptu space
    L number of layers
    H the width
    R output dim
    """
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(H, activation=act, input_shape=(P,)) if i == 0 else tf.keras.layers.Dense(H, activation=act) for i in range(L)] + 
        [tf.keras.layers.Dense(R)
        ])
    model.build(input_shape=[P])

    return model

def get_extent(design, model):
    image = model(tf.cast(design, tf.float32)).numpy()
    R = image.shape[1]
    extent = []
    for r in range(R):
        extent.append((min(image[:,r]), max(image[:,r])))
    return extent


def neural_plot(design, response, model, objective, figname = 'temp.pdf', B = 100000):
    P = design.shape[1]

    # Sample the objective at many points
    X_samp = np.random.uniform(size=[B,P])
    Zu_samp = model(tf.cast(X_samp, tf.float32))
    y_samp = np.apply_along_axis(objective, 1, X_samp)

    # Build a KNN response surface
    pred = KNeighborsRegressor(n_neighbors = 5)
    pred.fit(Zu_samp, y_samp)
    delta = 0.025
    x = np.arange(extent[0], extent[1], delta)
    y = np.arange(extent[2], extent[3], delta)
    X, Y = np.meshgrid(x, y)
    toplot = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            toplot[i,j] = pred.predict(np.array([X[i,j],Y[i,j]]).reshape(1,-1))

    design_tf = tf.Variable(design)
    Z_sol = model(tf.cast(design_tf, tf.float32)).numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(toplot, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=extent)
    plt.scatter(Z_sol[:,0], Z_sol[:,1], c = response)
    plt.autumn()
    plt.show()
    plt.savefig(figname)
