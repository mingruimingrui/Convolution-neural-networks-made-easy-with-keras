import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# gloabl variables
orig_shape = (1,32,32,3)
target_shape = (1,126,126,3)

# we first gather the original dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# then imgs are saved to an array for individual transformation
# this process can be quickened by doing the transormation by batch
# but that would take more computer memory, we are talking > 2GB of data
# feel free to edit this code to suit your envoirnmental requirements!
orig_imgs = []
for i in range(len(X_train)):
    orig_imgs.append(X_train[i])
del X_train
for i in range(len(X_test)):
    orig_imgs.append(X_test[i])
del X_test

# this method will take a single 32x32 image and transform it into a 126x126 img
# method used is simple gradient wise transformation
def image_transform(img1):
    img2 = np.zeros(target_shape[1:], dtype='uint8')
    for i in range(target_shape[1] - 1):
        for j in range(target_shape[2] - 1):
            xm = i % 4 / 4
            ym = j % 4 / 4
            if xm == 0:
                if ym == 0:
                    img2[i,j,:] = img1[i//4, j//4, :].astype('uint8')
                else:
                    img2[i,j,:] = (
                        img1[i//4, j//4, :] * (1-ym) +
                        img1[i//4, j//4+1, :] * ym
                    ).astype('uint8')
            else:
                if ym == 0:
                    img2[i,j,:] = (
                        img1[i//4, j//4, :] * (1-xm) +
                        img1[i//4+1, j//4, :] * xm
                    ).astype('uint8')
                else:
                    img2[i,j,:] = (
                        img1[i//4, j//4, :] * (1-ym) * (1-xm) +
                        img1[i//4, j//4+1, :] * ym * (1-xm) +
                        img1[i//4+1, j//4, :] * (1-ym) * xm +
                        img1[i//4+1, j//4+1, :] * ym * xm
                    ).astype('uint8')
    return img2

# transformed images are saved and a progress idicator is printed
transformed_imgs = []
for i in range(len(orig_imgs)):
    transformed_imgs.append(image_transform(orig_imgs[i]))
    if i % 600 == 0:
        print(str(i/600) + '%')

transformed_imgs = np.array(transformed_imgs)

X_train = transformed_imgs[:50000]
X_test = transformed_imgs[:50000]

# data is then stored into ./data
np.save('./data/X_train_126.npy', X_train)
np.save('./data/X_test_126.npy', X_test)
np.save('./data/y_train_126.npy', y_train)
np.save('./data/y_test_126.npy', y_test)
