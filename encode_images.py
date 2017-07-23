from pathlib import Path
import numpy as np
from keras.datasets import cifar10

# gloabl variables
orig_shape = (1,32,32,3)
target_shape = (1,64,64,3)

def save_X(X_train, X_test):
    print('Saving X')

    if ~Path('./data/X_train_64.npy').is_file():
        np.save('./data/X_train_64.npy', X_train)

    if ~Path('./data/X_test_64.npy').is_file():
        np.save('./data/X_test_64.npy', X_test)

def save_y(y_train, y_test):
    print('Saving y')

    if ~Path('./data/y_train_64.npy').is_file():
        np.save('./data/y_train_64.npy', y_train)

    if ~Path('./data/y_test_64.npy').is_file():
        np.save('./data/y_test_64.npy', y_test)

def get_dataset():
    print('Loading Dataset')

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, y_train, X_test, y_test

def get_formatted_dataset():
    X_train, y_train, X_test, y_test = get_dataset()

    save_y(y_train, y_test)

    del y_train, y_test

    print('Formatting data')
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
        (target_shape[1]+2,target_shape[2]+2,target_shape[3]),
        dtype='uint8'
    )
    for i in range(orig_shape[1]):
        for j in range(orig_shape[2]):
            transformed_img[2*i+1,2*j+1,:] = orig_img[i,j,:]
            transformed_img[2*i+1,2*j+2,:] = 0.5 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j+1,:] = 0.5 * orig_img[i,j,:]
            transformed_img[2*i+1,2*j,:] = 0.5 * orig_img[i,j,:]
            transformed_img[2*i,2*j+1,:] = 0.5 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j+2,:] = 0.25 * orig_img[i,j,:]
            transformed_img[2*i,2*j+2,:] = 0.25 * orig_img[i,j,:]
            transformed_img[2*i+2,2*j,:] = 0.25 * orig_img[i,j,:]
            transformed_img[2*i,2*j,:] = 0.25 * orig_img[i,j,:]

    transformed_img = transformed_img[1:target_shape[1]-1,1:target_shape[2]-1,:].astype('uint8')

    return transformed_img

def get_transformed_imgs():
    transformed_imgs = []

    if Path('./data/X_64.npy').is_file():
        print('Loading transformed images, continuing job')

        imgs = np.load('./data/X_64.npy')

        for i in range(len(imgs)):
            transformed_imgs.append(imgs[i])

    return transformed_imgs

def main():
    print('Welcome to img transformer')
    print('We shall transform our 32x32 imgs into 64x64')
    print('As dataset is really large, we segment this whole process into parts')
    print('Do note that we save our data every 10%')
    print('Feel free to pause and run this program in your free time!')

    transformed_imgs = get_transformed_imgs()

    nb_transformed = len(transformed_imgs)

    orig_imgs = get_formatted_dataset()
    orig_imgs = orig_imgs[len(transformed_imgs):]
    for i in range(len(orig_imgs)):
        transformed_img = image_transform(orig_imgs[i])
        transformed_imgs.append(transformed_img)

        if (i + nb_transformed) % 6000 == 0:
            print(str((i + nb_transformed) / 600) + '%')
            to_save = np.array(transformed_imgs)
            np.save('./data/X_64.npy', to_save)

    print('All images transformed')

    del orig_imgs

    transformed_imgs = np.array(transformed_imgs)

    X_train = transformed_imgs[:50000]
    X_test = transformed_imgs[:50000]

    save_X(X_train, X_test)

if __name__ == "__main__":
    # execute only if run as a script
    main()
