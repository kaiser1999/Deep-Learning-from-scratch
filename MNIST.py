import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from scratch import SGD, Adam, RMSprop, Adagrad, Momentum
from scratch import Flattening, Convolutional, Pooling, Transposed_Conv
from scratch import Dense
from scratch import Activation, Dropout, Batch_Normalization, Instance_Noramlization
from scratch import Sequential

#%%

np.random.seed(4012)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
N_train, N_test = len(X_train), len(X_test)
X_train, X_test = X_train.reshape(N_train, -1)/255, X_test.reshape(N_test, -1)/255
n_digit = len(np.unique(y_train))

model = Sequential()
model.add(Dropout(0.2, method="gaussian"))

model.add(Dense(64))
model.add(Activation("ReLU"))
model.add(Batch_Normalization("momentum"))
model.add(Dropout(0.5, method="gaussian"))

model.add(Dense(n_digit))
model.add(Activation("Softmax"))

model.compile(Adam(1e-2), loss="Categorical_Cross_Entropy")
model.train(X_train, y_train, bs=64, EPOCHS=30)

y_pred = model.predict(X_test)
print(np.mean(y_test == y_pred))
print(confusion_matrix(y_test, y_pred))

#%%

np.random.seed(4012)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_test = np.expand_dims(X_train, axis=-1)/255, np.expand_dims(X_test, axis=-1)/255
n_digit = len(np.unique(y_train))

model = Sequential()
model.add(Convolutional(n_f=32, s_f=[3, 3], stride=[1, 1]))
model.add(Instance_Noramlization())
model.add(Activation("ReLU"))
model.add(Pooling("MaxPooling", s_q=[5, 2]))

model.add(Convolutional(n_f=32, s_f=[3, 3], stride=[2, 2]))
model.add(Transposed_Conv(n_f=32, s_f=[3, 3], stride=[2, 2]))
model.add(Instance_Noramlization())
model.add(Activation("ReLU"))
model.add(Pooling("MaxPooling", s_q=[3, 2]))

model.add(Flattening())
model.add(Dense(32))
model.add(Activation("Sigmoid"))

model.add(Dense(n_digit))
model.add(Activation("Softmax"))

model.compile(Adam(1e-3), loss="Categorical_Cross_Entropy")
model.train(X_train, y_train, bs=64, EPOCHS=10)

y_pred = model.predict(X_test)
print(np.mean(y_test == y_pred))
print(confusion_matrix(y_test, y_pred))

#%%

np.random.seed(4012)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_test = np.expand_dims(X_train, axis=-1)/255, np.expand_dims(X_test, axis=-1)/255
n_digit = len(np.unique(y_train))

model = Sequential()
model.add(Convolutional(n_f=32, s_f=[3, 3], stride=[1, 1]))
model.add(Activation("ReLU"))
model.add(Pooling("MaxPooling", s_q=2))

model.add(Convolutional(n_f=32, s_f=[3, 3], stride=[1, 1]))
model.add(Activation("ReLU"))
model.add(Pooling("MaxPooling", s_q=2))

model.add(Flattening())
model.add(Dense(32))
model.add(Activation("Sigmoid"))

model.add(Dense(n_digit))
model.add(Activation("Softmax"))

model.compile(Adam(1e-3), loss="Categorical_Cross_Entropy")
model.train(X_train, y_train, bs=64, EPOCHS=10)

y_pred = model.predict(X_test)
print(np.mean(y_test == y_pred))
print(confusion_matrix(y_test, y_pred))