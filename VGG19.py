from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.activations import softmax
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.applications.vgg19 import preprocess_input, decode_predictions
from matplotlib import pyplot as plt
from keras.models import load_model

def VGG_19(weights_path):
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='block1_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block1_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',name='block2_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name="block2_pool"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',name='block3_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block3_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block4_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='block4_pool'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',name='block5_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name="block5_pool"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax',name='predictions'))

    
    model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    model = VGG_19("vgg19_weights_th_dim_ordering_th_kernels.h5")
    model.summary()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    img = cv2.resize(cv2.imread('truck.jpeg'), (224, 224))
    
    mean_pixel = [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)

   
    out = model.predict(img)
    # 
    print(decode_predictions(out,top=3))