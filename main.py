""" making any sort of working demo using keras tuner

    so then I can isolate what' snot working in my code by moving that working eample closer to my
    config.yml init method.
"""

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import tensorflow as tf
import tensorflow.keras.datasets.cifar10 as cifar10
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np

HEIGHT = 200
WIDTH = 200

def get_cifer10_dataset():
    """ just to play around """
    train, test = cifar10.load_data()
    x_train, y_train = train
    x_test, y_test = test
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def get_mnist_dataset():
    train, test = mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def prepare_data(training_data):

    # first preprocess the data
    # training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
    print(f"Shape: {training_data.shape},")
    print(f"Min: {np.min(training_data)},")
    print(f"Max: {np.max(training_data)},")
    cropper = CenterCrop(height=32, width=32)
    scale = 0.00392156862745098                 # or 1.0 /255
    scaler = Rescaling(scale=scale)
    training_data = scaler(cropper(training_data))
    print(f"Shape: {training_data.shape},")
    print(f"Min: {np.min(training_data)},")
    print(f"Max: {np.max(training_data)},")
    # print(training_data)

    return training_data

def build_model():
    inputs = keras.Input(shape=(HEIGHT, WIDTH, 3))

    # making things simple, I don't CenterCrop, just set the input to whatever I'm feeding it
    # I do rescale to 0-1 values
    x = Rescaling(scale=1.0 / 255)(inputs)

    # this basically the doc's architecture
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# fake data, easier to play with shapes  than with the actual CIFAR10 data to debug
# shape is (500, 200, 200, 3), cifar10's is (60000, 32, 32, 3)
data = np.random.randint(0, 255, size=(500, HEIGHT, WIDTH, 3)).astype("float32")
# shape is (500,1) cifar10's is (60000,1). Just 10 categories to match the output layer
labels = np.random.randint(0,9, size=(500,)).astype("int8")

print("got fake data... ")
model = build_model()
print("model built... ")
model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy())
print("model compile...")
model.fit(data, labels)
print("done")

#
# processed_data = model(data)
# print(processed_data.shape)
# dataset = tf.data.Dataset.from_tensor_slices((data, labels)).batch(20)
# model.fit(dataset)
