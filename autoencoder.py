from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Autoencoder:
    def __init__(self, input_dimension, activation="sigmoid", optimizer="adam", loss="mse"):
        # set bottleneck 1/3 of the input layer size
        input_dim = input_dimension[0]
        encoding_dim = int(np.floor(input_dim / 3))

        # input layer size= # of attributes in the dataset after one-hot encoding
        input_layer = layers.Input(shape=input_dimension)  # Input Layer

        encoded = layers.Dense(encoding_dim, activation=activation)(input_layer)  # Code Layer
        decoded = layers.Dense(input_dim, activation="linear")(encoded)  # Output Layer

        self._autoencoder = keras.Model(input_layer, decoded)

        self._autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, train_data, validation_data, epochs, batch_size=32, shuffle=True, ):
        return self._autoencoder.fit(train_data,
                                     train_data,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     validation_data=(validation_data, validation_data)).history

    def predict(self, data):
        return self._autoencoder.predict(data)