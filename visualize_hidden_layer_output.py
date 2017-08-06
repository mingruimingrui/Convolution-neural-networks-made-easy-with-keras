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
layer_depths = ['conv2d_1','conv2d_2','conv2d_3','conv2d_4']

labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

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
    if (Path('./models/convnet_improved_model.json').is_file() == False) | (Path('./models/convnet_improved_model.json').is_file() == False):
        sys.stdout.write('Please train model using basic_model.py first')
        sys.stdout.flush()
        raise SystemExit

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model

def remove_till_layer(model, layer_name):
    while model.layers[len(model.layers)-1].name != layer_name:
        model.pop()

    return model

def get_random_img(X, y):
    i = np.random.randint(0, len(X))
    img = X[i].reshape(X_shape)
    label_id = y[i].argmax()

    return img, label_id

def get_random_correct_img(X, y, model):
    found = False

    while found == False:
        img, label_id = get_random_img(X, y)
        pred = model.predict(img)[0]
        if pred.argmax() == label_id:
            found = True

    return img, label_id

def generate_conv_layer_models():
    sys.stdout.write('Generating layer models\n\n')
    sys.stdout.flush()
    conv_models = []

    for layer_name in layer_depths:
        conv_models.append(remove_till_layer(load_model(), layer_name))

    return conv_models

def plot_hidden_layers(model, img, title=None):
    to_visual = model.predict(img)
    to_visual = to_visual.reshape(to_visual.shape[1:])

    _ = plt.figure()
    if title:
        _ = plt.suptitle(title)
    sub_plot_height = math.ceil(np.sqrt(to_visual.shape[2]))
    for i in range(to_visual.shape[2]):
        ax = plt.subplot(sub_plot_height, sub_plot_height, i+1)
        _ = plt.axis('off')
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])
        _ = ax.set_aspect('equal')
        _ = plt.imshow(to_visual[:, :, i], cmap='inferno')

def gen_models_and_visualize(X, y, n_imgs=3):
    full_model = load_model()
    conv_models = generate_conv_layer_models()

    for i in range(n_imgs):
        img, label_id = get_random_correct_img(X, y, full_model)

        _ = plt.imshow(img.reshape(img.shape[1:]))
        _ = plt.title(labels[label_id])

        index = 0
        for model in conv_models:
            index += 1
            plot_hidden_layers(model, img, 'conv layer {}'.format(index))

        plt.show()

def main():
    _, _, X, y = get_dataset()
    gen_models_and_visualize(X, y)

if __name__ == "__main__":
    # execute only if run as a script
    main()
