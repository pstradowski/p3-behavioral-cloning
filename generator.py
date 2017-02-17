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
from math import ceil

def moving_average(a, n=3):
    # Moving average	
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


logs = []
input_dirs=['./data', './mydata']
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
steer_ma = moving_average(np.array(driving_log.steering), n = 8)
driving_log = driving_log.assign( steer_ma = pd.Series(steer_ma))
train, validation = train_test_split(driving_log, test_size = 0.05)

img_col = 64
img_row = 64

def preprocess(line, steering):
    correction = 0.25
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
    random_bright = .15+np.random.uniform()
    ret[:,:,2] =ret[:,:,2]*random_bright
    ret = cv2.cvtColor(ret,cv2.COLOR_HSV2RGB)
    # Random flip
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        ret = cv2.flip(ret,1)
        steering = -steering

    # Jitter by Vivek Yadaw
    rows, cols, _ = ret.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    ret = cv2.warpAffine(ret, transMat, (cols, rows))
    
  #  Cropping - horizon 50 pixels, hood 10 pixels
 #   No cropping on x-axis
    y_from = 60
    y_to = ret.shape[0]-20
    ret = ret[y_from:y_to]

    ret = cv2.resize(ret,(img_col, img_row), interpolation=cv2.INTER_AREA)
    return(ret, steering)

class eq_vivek:
    #Tribute to Vivek Yadaw for excellent idea of equalization :-)
    def __init__(self, dataset, ep_size = 20224, zero_width = 0.15):
        self.dataset =  dataset
        self.counter = 1
        self.ep_size = ep_size
        self.zero_width = zero_width
        
    def __next__(self):
        if self.counter < self.ep_size :
            pass_nr = 1
        else:
            pass_nr = ceil(self.counter/self.ep_size)
        self.threshold = 1/pass_nr
        iterate = True
        while iterate:
            idx = np.random.randint(len(self.dataset))
            line = self.dataset.iloc[idx]
            if abs(line['steering']) < self.zero_width:
                dice = np.random.uniform()
                if dice > self.threshold:
                    iterate = False
            else:
                iterate = False
        self.counter += 1
        return line
        

class equalizer:
    def __init__(self, dataset):
        self.dataset =  dataset
        self.dataset = self.dataset.assign(bin = pd.Series(pd.cut(self.dataset['steering'], 
        180, labels = False) ))
        self.full_bins = np.unique(self.dataset['bin'], return_counts = True)
        nmax = np.max(self.full_bins[1])
        self.boundaries = np.cumsum(nmax/self.full_bins[1])  
    def __next__(self):
        dice = np.random.rand() * self.boundaries[-1]
        idx = np.searchsorted(self.boundaries, dice)
        selected_bin = self.full_bins[0][idx]
        lines = self.dataset[self.dataset.bin == selected_bin]
        lidx = np.random.randint(len(lines))
        line = lines.iloc[lidx]
        return line

class uni_eq:
    def __init__(self, dataset, n_bins = 8):
        self.dataset = dataset
        self.dataset = self.dataset.assign(bin = pd.Series(pd.cut(self.dataset['steering'], 
        n_bins, labels = False) ))
        self.bins = np.unique(self.dataset['bin'])

    def __next__(self):
        idx = np.random.randint(len(self.bins))
        basket = self.bins[idx]
        lines = self.dataset[self.dataset.bin == basket]
        lidx = np.random.randint(len(lines))
        line = lines.iloc[lidx]
        return line

class randliner:
    def __init__(self, dataset, direct_threshold = 0.35):
        epsilon = 0.1
        self.direct = dataset[abs(dataset.steering) < epsilon]
        self.turns = dataset[abs(dataset.steering) > epsilon]
        self.direct_threshold = direct_threshold
    def __next__(self):
        dice = np.random.rand()
        if dice < self.direct_threshold:
            direction = self.direct
        else:
            direction = self.turns
        idx = np.random.randint(len(direction))
        line = direction.iloc[idx]
        return line

def gen_train(train, batch_size=4):
    feeder = eq_vivek(train)
    while 1:
        X_train = np.zeros((batch_size, img_row, img_col, 3))
        y_train = np.zeros(batch_size)
        for i in range(batch_size):
            line = next(feeder)
            y = line['steering']
            X, y = preprocess(line, y)
            X_train[i] = X
            y_train[i] = y
        yield(X_train, y_train)

def gen_valid(val_set, batch_size = 1):
    
    feeder = eq_vivek(val_set)
    while 1:
        X_valid = np.zeros((batch_size, img_row, img_col, 3))
        y_valid = np.zeros(batch_size)
        for i in range(batch_size):
            curr_line = next(feeder)
            img = cv2.imread(curr_line['center'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_valid[i] = cv2.resize(img, (img_col, img_row), interpolation=cv2.INTER_AREA)
            y_valid[i] = curr_line['steering']
        yield(X_valid, y_valid)

def nvidia_model(img_channels=3, dropout = 0.6):
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
if __name__ == "__main__":
    model = nvidia_model()
    model_name='vivek'
    checkpointer =  ModelCheckpoint(filepath= 'models/' + 
        model_name + "{epoch:02d}-{val_loss:.2f}.hdf5", 
        verbose=1, save_best_only=False)

    model.fit_generator(gen_train(train, batch_size = 128),
    samples_per_epoch = 20224,
    nb_epoch = 10,
    validation_data = gen_valid(validation) ,
    nb_val_samples = len(validation),
    callbacks=[checkpointer])

    model.save('models/' + model_name + ".hdf5")
