import sys
from pathlib import Path
import numpy as np
from keras.datasets import cifar10

# gloabl variables
orig_shape = (1,32,32,3)
target_shape = (1,64,64,3)

def save_X(X_train, X_test):
    sys.stdout.write('Saving X_data\n\n')
    sys.stdout.flush()

    if ~Path('./data/X_train_64.npy').is_file():
        np.save('./data/X_train_64.npy', X_train)

    if ~Path('./data/X_test_64.npy').is_file():
        np.save('./data/X_test_64.npy', X_test)

def save_y(y_train, y_test):
    sys.stdout.write('Saving y_data\n\n')
    sys.stdout.flush()

    if ~Path('./data/y_train_64.npy').is_file():
        np.save('./data/y_train_64.npy', y_train)

    if ~Path('./data/y_test_64.npy').is_file():
        np.save('./data/y_test_64.npy', y_test)

def get_dataset():
    sys.stdout.write('Loading Dataset\n')
    sys.stdout.flush()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, y_train, X_test, y_test

def get_formatted_dataset():
    X_train, y_train, X_test, y_test = get_dataset()

    save_y(y_train, y_test)

    del y_train, y_test

    sys.stdout.write('Formatting data\n')
    sys.stdout.flush()
    # this is probably something I should fix
    # not all these data needs to be formatted

    orig_imgs=[]
    for i in range(len(X_train)):
        orig_imgs.append(X_train[i])
    for i in range(len(X_test)):
        orig_imgs.append(X_test[i])
    del X_train, X_test

    return orig_imgs

def image_transform(orig_img):
    transformed_img = np.zeros(
        (target_shape[1]+2,target_shape[2]+2,target_shape[3]))
    for i in range(orig_shape[1]):
        for j in range(orig_shape[2]):
            transformed_img[2*i+1,2*j+1,:] += orig_img[i,j,:]
            transformed_img[2*i+1,2*j+2,:] += 0.5 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j+1,:] += 0.5 * orig_img[i,j,:]
            transformed_img[2*i+1,2*j,:] += 0.5 * orig_img[i,j,:]
            transformed_img[2*i,2*j+1,:] += 0.5 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j+2,:] += 0.25 * orig_img[i,j,:]
            transformed_img[2*i,2*j+2,:] += 0.25 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j,:] += 0.25 * orig_img[i,j,:]
            transformed_img[2*i,2*j,:] += 0.25 * orig_img[i,j,:]

    transformed_img = transformed_img[1:target_shape[1]+1,1:target_shape[2]+1,:].astype('uint8')

    return transformed_img

def get_transformed_imgs():
    transformed_imgs = []

    if Path('./data/X_64.npy').is_file():
        sys.stdout.write('Loading transformed images, continuing job\n')
        sys.stdout.flush()

        imgs = np.load('./data/X_64.npy')

        sys.stdout.write('Previously at {}%\n\n'.format(int(len(imgs)/600)))
        sys.stdout.flush()

        for i in range(len(imgs)):
            transformed_imgs.append(imgs[i])

    return transformed_imgs

def main():
    sys.stdout.write('Welcome to image transformer\n\n')
    sys.stdout.write('We shall transform our 32x32 imgs into 64x64\n')
    sys.stdout.write('As dataset is really large, we segment this whole process into parts\n')
    sys.stdout.write('Do note that we save our data every 10%\n')
    sys.stdout.write('Feel free to pause and run this program in your free time!\n\n')
    sys.stdout.flush()

    transformed_imgs = get_transformed_imgs()

    nb_transformed = len(transformed_imgs)

    orig_imgs = get_formatted_dataset()
    orig_imgs = orig_imgs[len(transformed_imgs):]
    for i in range(len(orig_imgs)):
        transformed_img = image_transform(orig_imgs[i])
        transformed_imgs.append(transformed_img)

        if i+nb_transformed == 60000:
            to_save = np.array(transformed_imgs)
            np.save('./data/X_64.npy', to_save)
        elif ((i + nb_transformed) % 6000 == 0) and (i + nb_transformed != 0):
            sys.stdout.write(str((i + nb_transformed) / 600) + '%\n')
            sys.stdout.flush()
            sys.stdout.write('\b\b\b\b\b\b')
            to_save = np.array(transformed_imgs)
            np.save('./data/X_64.npy', to_save)

    sys.stdout.write('All images transformed\n')
    sys.stdout.flush()

    del orig_imgs

    transformed_imgs = np.array(transformed_imgs)

    X_train = transformed_imgs[:50000]
    X_test = transformed_imgs[50000:]

    save_X(X_train, X_test)

if __name__ == "__main__":
    # execute only if run as a script
    main()
