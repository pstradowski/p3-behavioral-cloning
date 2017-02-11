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
from os import path
from keras.callbacks import ModelCheckpoint


logs = []
input_dirs=['./data']
col_names = ('center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed')

for dir in input_dirs:
    csv_file = dir + '/driving_log.csv'
    log = pd.read_csv(csv_file, index_col = False, 
        sep = '\,\s*', engine = 'python', names = col_names)
    for i in ('left', 'center', 'right'):
        log[i] = log[i].str.rsplit("/", n=1).str[-1].apply(lambda p: path.join(dir+'/IMG', p))
    for i in ('steering', 'throttle', 'brake', 'speed'):
        log[i] = log[i].astype(np.float32)
    logs.append(log)

driving_log = pd.concat(logs, axis=0, ignore_index=True)

nonzero = driving_log[driving_log.steering != 0]

train, validation = train_test_split(nonzero, test_size = 0.2)
#train, test = train_test_split(t1, test_size = 0.25)

img_col = 64
img_row = 64

def preprocess(line, steering):
    correction = 0.15
    coin = np.random.randint(0, 3)
    camera = 'center'
    if coin == 1: 
        camera = 'left'
        steering += correction
    elif coin == 2:
        camera = 'right'
        steering -= correction

    img = cv2.imread(line[camera])
    # Random brightness set
    # as proposed by ViVek Yadav
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
    ret = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    ret[:,:,2] =ret[:,:,2]*random_bright
    ret = cv2.cvtColor(ret,cv2.COLOR_HSV2RGB)
    # Random flip
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        ret = cv2.flip(ret,1)
        steering = -steering
        
    # Jitter by Vivek Yadaw
    rows, cols, _ = ret.shape
    transRange = 150
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    ret = cv2.warpAffine(ret, transMat, (cols, rows))
    
    # Cropping - horizon 50 pixels, hood 10 pixels
    # No cropping on x-axis
    y_from = 40
    y_to = ret.shape[1]-10
    ret = ret[y_from:y_to]

    ret = cv2.resize(ret,(img_col, img_row), interpolation=cv2.INTER_AREA)
    return(ret, steering)

def gen_train(train, batch_size = 4):
    train = train.assign(bin = pd.Series(pd.cut(train['steering'], 180, labels = False) ))
    full_bins = np.unique(train['bin'])
    X_train = np.zeros((batch_size, img_row, img_col, 3))
    y_train = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            selected_bin = full_bins[np.random.randint(len(full_bins))]
            bin_lines = train[train.bin == selected_bin]
            idx = np.random.randint(len(bin_lines))
            line = bin_lines.iloc[idx]
            y = line['steering']
            X, y = preprocess(line, y)
            X_train[i] = X
            y_train[i] = y

        yield(X_train, y_train)

def gen_valid(val_set, batch_size = 1):
    X_valid = np.zeros((batch_size, img_row, img_col, 3))
    y_valid = np.zeros(batch_size)
    line = 0
    while 1:
        for i in range(batch_size):
            curr_line = val_set.iloc[line]
            img = cv2.imread(curr_line['center'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_valid[i] = cv2.resize(img, (img_col, img_row), interpolation=cv2.INTER_AREA)
            y_valid[i] = curr_line['steering']
            if line < len(val_set) - 1:
                line += 1
            else:
                line = 0
        yield(X_valid, y_valid)


def comma_model(time_len=1):
    ch, row, col = 3, img_row, img_col  # camera format

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

def nvidia_model(img_channels=3, dropout=.5):
    img_height = img_row
    img_width = img_col
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

    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

model = nvidia_model()
model_name='nonzero_udacity'
checkpointer =  ModelCheckpoint(filepath= 'models/' + 
    model_name + "{epoch:02d}-{val_loss:.2f}.hdf5", 
    verbose=1, save_best_only=True)

model.fit_generator(gen_train(train, batch_size = 128),
samples_per_epoch = 20224,
nb_epoch = 20,
validation_data = gen_valid(validation) ,
nb_val_samples = len(validation),
callbacks=[checkpointer])

model.save('models/' + model_name + ".hdf5")
