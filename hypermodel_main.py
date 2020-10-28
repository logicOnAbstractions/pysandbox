
from kerastuner import HyperModel
import keras
from kerastuner.tuners import RandomSearch
import tensorflow.keras.datasets.mnist as mnist

def get_mnist_dataset():
    """ just to play around """
    train, test = mnist.load_data(path="mnist.npz")
    x_train, y_train = train
    x_test, y_test = test
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


class Architecture(HyperModel):
    def __init__(self, *args, **kwargs):
        super(Architecture, self).__init__(*args, **kwargs)

    def build(self, hp):
        input_lay = keras.layers.Input(shape=(28, 28))
        flat = keras.layers.Flatten(input_shape=(28, 28))(input_lay)

        print(f"dims with functiona pi build: {flat}")
        hp_units = hp.Int(name='units', min_value=32, max_value=512, step=32)
        middle_lay1 = keras.layers.Dense(units=hp_units, activation="relu")(flat)
        ouput_lay = keras.layers.Dense(units=10, activation="softmax")(middle_lay1)

        model = keras.Model(input_lay, ouput_lay)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

hypermodel = Architecture()
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='helloworld')

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
