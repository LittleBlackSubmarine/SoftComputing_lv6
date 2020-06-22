from keras import layers
from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical

WindowSize = [(3, 3), (5, 5), (7, 7)]
Activation = ["relu", "tanh", "sigmoid"]

### Data preparing ###
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

for win_size in WindowSize:
        for activ_fn in Activation:

            ### Model 1 ###
            model_1 = models.Sequential()
            model_1.add(layers.Conv2D(64, win_size, activation=activ_fn, input_shape=(28, 28, 1)))
            model_1.add(layers.Conv2D(32, win_size, activation=activ_fn))
            model_1.add(layers.Flatten())
            model_1.add(layers.Dense(10, activation='softmax'))

            model_1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            model_1.summary()

            ### Model 2###
            model_2 = models.Sequential()
            model_2.add(layers.Conv2D(64, win_size, activation=activ_fn, input_shape=(28, 28, 1)))
            model_2.add(layers.MaxPooling2D((2, 2)))
            model_2.add(layers.Conv2D(32, win_size, activation=activ_fn))
            model_2.add(layers.MaxPooling2D((2, 2)))
            model_2.add(layers.Flatten())
            model_2.add(layers.Dense(10, activation='softmax'))

            model_2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            model_2.summary()

            ### Training ###

            model_1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=2)
            model_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=2)
            print(" Window size:" + str(win_size) + ", Activation: " + activ_fn + "\n\n" + 30 * "*" + "\n\n")



