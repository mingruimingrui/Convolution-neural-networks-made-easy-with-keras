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
layer_depths = ['conv2d_1','conv2d_2','conv2d_3','conv2d_4','conv2d_5','conv2d_6']

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

def get_random_img(X, y):
    i = np.random.randint(0, len(X))
    img = X[i].reshape(X_shape)
    label = labels[y[i].argmax()]

    return img, label

def get_random_correct_img(X, y, model):
    found = False

    while found == False:
        img, label = get_random_img(X, y)
        pred = model.predict(img)[0]
        if pred.argmax() == y[i].argmax()):
            found = True

    return img, label

def visualize(X, y, model, n_imgs=3):
    for i in range(n_imgs):
        img = get_random_correct_img()



def main():
    _, _, X, y = get_dataset()
    model = load_model()
    visualize(X, y, model, n_imgs=3)

if __name__ == "__main__":
    # execute only if run as a script
    main()
