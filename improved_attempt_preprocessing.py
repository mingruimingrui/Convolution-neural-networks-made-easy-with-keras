import sys
import numpy as np
import keras
from pathlib import Path

# global static variables
dtype_mult = 255.0 # unit8
num_classes = 10

def check_pre_req():
    if (
        Path('./data/X_train_64.npy').is_file() and
        Path('./data/y_train_64.npy').is_file() and
        Path('./data/X_test_64.npy').is_file() and
        Path('./data/y_test_64.npy').is_file()
    ) == False:
        sys.stdout.write('Please complete the execution of encode_images.py first!\n')
        sys.stdout.flush()
        raise SystemExit


def get_preprocessed_dataset():
    X_train, y_train, X_test, y_test = get_dataset()

    sys.stdout.write('Preprocessing Dataset\n')
    sys.stdout.flush()

    X_train = X_train.astype('float32') / dtype_mult
    X_test = X_test.astype('float32') / dtype_mult
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def preprocess_X_train():
    sys.stdout.write('Preprocessing X train\n')

    if Path('./data/X_train_64_preprocessed.npy').is_file() == False:
        X_train = np.load('./data/X_train_64.npy')
        X_train = X_train.astype('float32') / dtype_mult
        np.save('./data/X_train_64_preprocessed.npy', X_train)

        del X_train

def preprocess_y_train():
    sys.stdout.write('Preprocessing y train\n')
    sys.stdout.flush()

    if Path('./data/y_train_64_preprocessed.npy').is_file() == False:
        y_train = np.load('./data/y_train_64.npy')
        y_train = keras.utils.to_categorical(y_train, num_classes)
        np.save('./data/y_train_64_preprocessed.npy', y_train)

        del y_train

def preprocess_X_test():
    sys.stdout.write('Preprocessing X test\n')
    sys.stdout.flush()

    if Path('./data/X_test_64_preprocessed.npy').is_file() == False:
        X_test = np.load('./data/X_test_64.npy')
        X_test = X_test.astype('float32') / dtype_mult
        np.save('./data/X_test_64_preprocessed.npy', X_test)

        del X_test

def preprocess_y_test():
    sys.stdout.write('Preprocessing y test\n')
    sys.stdout.flush()

    if Path('./data/y_test_64_preprocessed.npy').is_file() == False:
        y_test = np.load('./data/y_test_64.npy')
        y_test = keras.utils.to_categorical(y_test, num_classes)
        np.save('./data/y_test_64_preprocessed.npy', y_test)

        del y_test

def main():
    check_pre_req()
    preprocess_X_train()
    preprocess_y_train()
    preprocess_X_test()
    preprocess_y_test()
    sys.stdout.write('All preprocessing done\n')
    sys.stdout.flush()

if __name__ == "__main__":
    # execute only if run as a script
    main()
