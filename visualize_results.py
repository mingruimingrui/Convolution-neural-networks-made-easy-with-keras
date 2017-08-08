import sys
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

model_path = './models/stashed/convnet_model.json'
weight_path = './models/stashed/convnet_weights.h5'
dtype_mult = 255
num_classes = 10
X_shape = (-1,32,32,3)

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

def print_accuracy(model, X_train, y_train, X_test=None, y_test=None):
    print('Calculating model accuracy')

    pred_train = model.predict(X_train)
    acc_train = np.sum(pred_train.argmax(axis=1) == y_train.argmax(axis=1)) / len(y_train)
    print('Training acc: {}'.format(acc_train))

    if (type(X_test) != type(None)) & (type(y_test) != type(None)):
        pred_test = model.predict(X_test)
        acc_test = np.sum(pred_test.argmax(axis=1) == y_test.argmax(axis=1)) / len(y_test)
        print('Testing acc: {}\n'.format(acc_test))

    sys.stdout.flush()

def get_random_img(X, y):
    i = np.random.randint(0, len(X))
    img = X[i].reshape(X_shape)
    label = labels[y[i].argmax()]

    return img, label

def visualize_examples(X, y, model, n_imgs=3):
    for i in range(n_imgs):
        img, label = get_random_img(X, y)
        pred = model.predict(img).squeeze()
        pred = list(map(lambda x: [labels[x], pred[x]], labels))
        pred.sort(key=lambda x: x[1], reverse=True)

        print('Top 3 likely predictions:')
        print(pred[:3])
        print()
        sys.stdout.flush()

        _ = plt.imshow(img.squeeze())
        _ = plt.title('Actual label: {}'.format(label))
        _ = plt.show()

    sys.stdout.write('\n')

def print_error_breakdown(X, y, model):
    table = np.zeros((num_classes, num_classes))
    pred = model.predict(X)

    for i in range(len(pred)):
        table[y[i].argmax(), pred[i].argmax()] += 1

    table = pd.DataFrame(table.astype('int'))
    table.index = labels.values()
    table.columns = labels.values()
    print(table)
    sys.stdout.write('\n')
    sys.stdout.flush()

def main():
    X_train, y_train, X_test, y_test = get_dataset()
    model = load_model()
    print_accuracy(model, X_train, y_train, X_test, y_test)
    visualize_examples(X_test, y_test, model, n_imgs=3)
    print('Training error breakdown')
    print_error_breakdown(X_train, y_train, model)
    print('Testing error breakdown')
    print_error_breakdown(X_test, y_test, model)

if __name__ == "__main__":
    # execute only if run as a script
    main()
