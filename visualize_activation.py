import sys
import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

model_path = './models/stashed/convnet_model.json'
weight_path = './models/stashed/convnet_weights.h5'
dtype_mult = 255
num_classes = 10
X_shape = (-1,32,32,3)
layer_depths = [2,4,8,10]

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_dataset():
    sys.stdout.write('Loading Dataset\n\n')
    sys.stdout.flush()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # we perform a series of normalization and binarizer on the dataset here
    X_train = X_train.astype('float32') / dtype_mult
    X_test = X_test.astype('float32') / dtype_mult
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def load_model():
    sys.stdout.write('Loading model\n\n')
    sys.stdout.flush()

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model

def remove_till_layer(model, layer):
    while len(model.layers) > layer:
        model.pop()

    return model

def get_random_img(X):

    return X[np.random.randint(0, len(X))].reshape(X_shape)

def generate_conv_layer_models():
    sys.stdout.write('Generating layer models\n\n')
    sys.stdout.flush()
    conv_models = []

    for i in layer_depths:
        conv_models.append(remove_till_layer(load_model(), i))

    return conv_models

def plot_hidden_layers(model, img):
    to_visual = model.predict(img)
    to_visual = to_visual.reshape(to_visual.shape[1:])

    _ = plt.figure()
    sub_plot_height = math.ceil(np.sqrt(to_visual.shape[2]))
    for i in range(to_visual.shape[2]):
        ax = plt.subplot(sub_plot_height, sub_plot_height, i+1)
        _ = plt.axis('off')
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])
        _ = ax.set_aspect('equal')
        _ = plt.imshow(to_visual[:, :, i])

def visualize(X, conv_models, n_imgs=10):
    for i in range(n_imgs):
        img = get_random_img(X)

        _ = plt.imshow(img.reshape(img.shape[1:]))

        for model in conv_models:
            plot_hidden_layers(model, img)

        plt.show()

def main():
    X, y, _, _ = get_dataset()
    conv_models = generate_conv_layer_models()
    visualize(X, conv_models)

main()
