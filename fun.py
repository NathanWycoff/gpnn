from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

outputTensor = model.output #Or model.layers[index].output

listOfVariableTensors = model.trainable_weights

trainingExample = np.random.random((1,8))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
