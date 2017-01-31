import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import json


csv_file = 'data/driving_log.csv'
driving_log = pd.read_csv(csv_file, index_col = False, sep = '\,\s*')
t1, validation = train_test_split(driving_log, test_size = 0.2)
train, test = train_test_split(t1, test_size = 0.25)

direct = train[train['steering'] == 0]
turns = train[train['steering'] != 0]
direct = direct.sample(frac=0.5).reset_index(drop = True)
balanced_train = pd.concat([direct, turns])
balanced_train = balanced_train.sample(frac=1).reset_index(drop = True)
img_path = 'data/'

def preprocess_img(img, col=128, row=64):
    ret = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    ret[:,:,2] =ret[:,:,2]*random_bright
    ret = cv2.cvtColor(ret,cv2.COLOR_HSV2RGB)
    ret = cv2.resize(ret,(col, row), interpolation=cv2.INTER_AREA)
    return(ret)

def gen_train(train_set, batch_size = 4):
    col = 128
    row = 64
    X_train = np.zeros((batch_size, row, col, 3))
    y_train = np.zeros(batch_size)
    imgs_per_line = 2
    line = 0
    while 1:
        for i in range(int(batch_size/imgs_per_line)):
            curr_line = train_set.iloc[line]
            center = cv2.imread(img_path + curr_line['center'])
            left = cv2.imread(img_path + curr_line['left'])
            right = cv2.imread(img_path + curr_line['right'])
            center = preprocess_img(center)
            left = preprocess_img(left)
            right = preprocess_img(right)
            mirror = cv2.flip(center, 1)
            steering = curr_line['steering']
            X_train[i] = center
            y_train[i] = steering 
            X_train[i+1] = left
            y_train[i+1] = steering + 0.25        
            X_train[i+2] = right
            y_train[i+2] = steering - 0.25
            X_train[i+3] = mirror
            y_train[i+3] = -steering
            if line < len(train_set) - 1:
                line +=1
            else:
                line = 0
        yield(X_train, y_train)

def gen_valid(val_set, batch_size = 4,
    col=128, row=64):
    X_valid = np.zeros((batch_size, row, col, 3))
    y_valid = np.zeros(batch_size)
    line = 0
    while 1:
        for i in range(batch_size):
            curr_line = val_set.iloc[line]
            img = cv2.imread(img_path + curr_line['center'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_valid[i] = cv2.resize(img, (col, row), interpolation=cv2.INTER_AREA)
            y_valid[i] = curr_line['steering']
            if line < len(val_set) - 1:
                line += 1
            else:
                line = 0
        yield(X_valid, y_valid)


def comma_model(time_len=1):
    ch, row, col = 3, 64, 128  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch ),
            output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def nvidia_model(img_height=64, img_width=128, img_channels=3,
                       dropout=.4):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    return model

model = nvidia_model()
model.fit_generator(gen_train(balanced_train, batch_size = 128),
samples_per_epoch = len(balanced_train)*4,
nb_epoch = 5,
validation_data = gen_valid(validation) ,
nb_val_samples = len(validation))
model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)
