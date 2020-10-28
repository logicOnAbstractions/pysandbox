""" making any sort of working demo using keras tuner

    so then I can isolate what' snot working in my code by moving that working eample closer to my
    config.yml init method.
"""

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import tensorflow.keras.datasets.mnist as mnist


def get_mnist_dataset():
    """ just to play around """
    train, test = mnist.load_data(path="mnist.npz")
    x_train, y_train = train
    x_test, y_test = test
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    print(f"dims with Sequential() build: {model.output_shape}")

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model



def model_builder_functional(hp):

    input_lay = keras.layers.Input(shape=(28,28))
    flat = keras.layers.Flatten(input_shape=(28,28))(input_lay)

    print(f"dims with functiona pi build: {flat}")
    hp_units = hp.Int(name='units', min_value=32, max_value=512, step=32)
    middle_lay1 = keras.layers.Dense(units=hp_units, activation="relu")(flat)
    ouput_lay   = keras.layers.Dense(units=10, activation="softmax")(middle_lay1)

    model = keras.Model(input_lay, ouput_lay)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def model_builder_functional_test(hp):

    input_lay = keras.layers.Input(shape=(28,28))
    flat = keras.layers.Flatten(input_shape=(28,28))(input_lay)

    print(f"dims with functiona pi build: {flat}")
    # hp_units = hp.Int(name='units', min_value=32, max_value=512, step=32)
    hp_units = hp.Int(name='units', min_value=32, max_value=512, step=32)


    middle_lay1 = keras.layers.Dense(units=hp_units, activation="relu")(flat)
    ouput_lay   = keras.layers.Dense(units=10, activation="softmax")(middle_lay1)

    model = keras.Model(input_lay, ouput_lay)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
#
# tuner = RandomSearch(model_builder, objective='val_accuracy', max_trials=5, executions_per_trial=2, directory='my_dir', project_name='helloworld')
# tuner = RandomSearch(model_builder_functional, objective='val_accuracy', max_trials=5, executions_per_trial=2, directory='my_dir', project_name='helloworld')
tuner = RandomSearch(model_builder_functional_test, objective='val_accuracy', max_trials=5, executions_per_trial=2, directory='my_dir', project_name='helloworld')



# You can print a summary of the search space:
tuner.search_space_summary()

# ok next we get data, let's use the MNIST one for simplicity, as we already do in our code
data_dict = get_mnist_dataset()
x = data_dict["x_train"]
y = data_dict["y_train"]
x_val = data_dict["x_test"]
y_val = data_dict["y_test"]

print(f"label shapes: {y.shape} {y_val.shape}")
# call the tuner on that

tuner.search(x, y, epochs=2, validation_data=(x_val, y_val))
models = tuner.get_best_models(num_models=2)
tuner.results_summary()

