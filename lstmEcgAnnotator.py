# import matplotlib with pdf as backend
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import wfdb
import os
import numpy as np
import math
import sys
import scipy.stats as st
import glob, os
from os.path import basename

import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input, MaxPooling1D, concatenate
from keras.layers import LSTM, Bidirectional, Conv1D, Conv1D, Flatten, UpSampling1D
from keras.layers import Cropping1D, ZeroPadding1D
from keras.models import Sequential, load_model, Model
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF

np.random.seed(0)

annot_type = "pu"


def my_get_ecg_data(datfile):
    ## convert .dat/q1c to numpy arrays
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.

    annotator = annot_type
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0, sampto=None, pb_dir=None)  # read annotation
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot)  # annot start
    LastLstannot = max(i[0] for i in Lstannot) - 1  # annot end
    print("first-last annotation:", FirstLstannot, LastLstannot)

    record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot, sampto=LastLstannot)  # read signal
    annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,
                            sampto=LastLstannot)  ## get annotation between first and last.
    # this make annotation from 0 to end
    annotation2 = wfdb.Annotation(record_name=recordname, extension=annot_type,
                                  sample=(annotation.sample - FirstLstannot),
                                  symbol=annotation.symbol, aux_note=annotation.aux_note)

    Vctrecord = np.transpose(record[0])
    VctAnnotationHot = np.zeros((6, len(Vctrecord[1])),
                                dtype=np.int)  # 6 annotation type: 0: P, 1: PQ, 2: QR, 3: RS, 4: ST, 5: ISO (TP)
    #VctAnnotationHot[3] = 1  ## inverse of the others
    # print("ecg, 2 lead of shape" , Vctrecord.shape)
    # print("VctAnnotationHot of shape" , VctAnnotationHot.shape)
    # print('plotting extracted signal with annotation')
    # wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations = list(zip(annotation2.sample, annotation2.symbol))  ## zip coordinates + annotations (N),(t) etc)
    # print(VctAnnotations)
    annsamp = []
    anntype = []
    for i in range(len(VctAnnotations)):
        annsamp.append(VctAnnotations[i][0])
        anntype.append(VctAnnotations[i][1])
    length = len(annsamp)
    begin = -1
    end = -1
    s = 'none'
    states = {'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
    annot_expand = -1 * np.ones(length)

    for j, samp in enumerate(annsamp):
        if anntype[j] == '(':
            begin = samp
            if (end > 0) & (s != 'none'):
                if s == 'N':
                    VctAnnotationHot[1][end:begin] = 1
                    #annot_expand[end:begin] = states['st']
                elif s == 't':
                    VctAnnotationHot[3][end:begin] = 1
                    #annot_expand[end:begin] = states['iso']
                elif s == 'p':
                    VctAnnotationHot[5][end:begin] = 1
                    #annot_expand[end:begin] = states['pq']
        elif anntype[j] == ')':
            end = samp
            if (begin >= 0) & (s != 'none'):
                VctAnnotationHot[states[s]][begin:end] = 1
                #annot_expand[begin:end] = states[s]
        else:
            s = anntype[j]
    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot


# functions
def get_ecg_data(datfile):
    ## convert .dat/q1c to numpy arrays
    recordname = os.path.basename(datfile).split(".dat")[0]
    recordpath = os.path.dirname(datfile)
    cwd = os.getcwd()
    os.chdir(recordpath)  ## somehow it only works if you chdir.

    annotator = annot_type
    annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0, sampto=None, pb_dir=None)  # read annotation
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot)  # annot start
    LastLstannot = max(i[0] for i in Lstannot) - 1  # annot end
    print("first-last annotation:", FirstLstannot, LastLstannot)

    record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot, sampto=LastLstannot)  # read signal
    annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,
                            sampto=LastLstannot)  ## get annotation between first and last.
    # this make annotation from 0 to end
    annotation2 = wfdb.Annotation(record_name=recordname, extension=annot_type,
                                  sample=(annotation.sample - FirstLstannot),
                                  symbol=annotation.symbol, aux_note=annotation.aux_note)

    Vctrecord = np.transpose(record[0])
    VctAnnotationHot = np.zeros((6, len(Vctrecord[1])), dtype=np.int)
    # TODO: default annotation:
    # VctAnnotationHot[5] = 1  ## inverse of the others
    # TODO: my annotation:
    VctAnnotationHot[3] = 1
    # print("ecg, 2 lead of shape" , Vctrecord.shape)
    # print("VctAnnotationHot of shape" , VctAnnotationHot.shape)
    # print('plotting extracted signal with annotation')
    # wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations = list(zip(annotation2.sample, annotation2.symbol))  ## zip coordinates + annotations (N),(t) etc)
    # print(VctAnnotations)
    for i in range(len(VctAnnotations)):
        # print(VctAnnotations[i]) # Print to display annotations of an ecg
        try:

            if VctAnnotations[i][1] == "p":
                if VctAnnotations[i - 1][1] == "(":
                    pstart = VctAnnotations[i - 1][0]
                if VctAnnotations[i + 1][1] == ")":
                    pend = VctAnnotations[i + 1][0]
                if VctAnnotations[i + 3][1] == "N":
                    rpos = VctAnnotations[i + 3][0]
                    if VctAnnotations[i + 2][1] == "(":
                        qpos = VctAnnotations[i + 2][0]
                    if VctAnnotations[i + 4][1] == ")":
                        spos = VctAnnotations[i + 4][0]
                    for ii in range(0, 8):  ## search for t (sometimes the "(" for the t  is missing  )
                        if VctAnnotations[i + ii][1] == "t":
                            tpos = VctAnnotations[i + ii][0]
                            if VctAnnotations[i + ii + 1][1] == ")":
                                tendpos = VctAnnotations[i + ii + 1][0]
                                # print(ppos,qpos,rpos,spos,tendpos)

                                # TODO: default annotation:
                                """{'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
                                # P segment
                                VctAnnotationHot[0][pstart:pend] = 1
                                # part "nothing" between P and Q, previously left unnanotated, but categorical probably can't deal with that
                                VctAnnotationHot[1][pend:qpos] = 1
                                # QR
                                VctAnnotationHot[2][qpos:rpos] = 1
                                # RS
                                VctAnnotationHot[3][rpos:spos] = 1
                                # ST (from end of S to end of T)
                                VctAnnotationHot[4][spos:tendpos] = 1
                                # tendpos:pstart becomes 1, because it is inverted above
                                VctAnnotationHot[5][pstart:tendpos] = 0
                                #"""
                                # TODO: my annotation:
                                # """{'N': 0, 'st': 1, 't': 2, 'iso': 3, 'p': 4, 'pq': 5}
                                # QRS segment
                                VctAnnotationHot[0][qpos:spos] = 1
                                # ST segment
                                VctAnnotationHot[1][spos:tpos] = 1
                                # T
                                VctAnnotationHot[2][tpos:tendpos] = 1
                                # ISO
                                VctAnnotationHot[3][pstart:tendpos] = 0
                                # P
                                VctAnnotationHot[4][pstart:pend] = 1
                                # PQ
                                VctAnnotationHot[5][pend:qpos] = 1
                                # """


        except IndexError:
            pass

    Vctrecord = np.transpose(Vctrecord)  # transpose to (timesteps,feat)
    VctAnnotationHot = np.transpose(VctAnnotationHot)
    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot


# разделияет последовательность x на участки длинной и с перекрытием в o с соседом слева и справа
def splitseq(x, n, o):
    # split seq; should be optimized so that remove_seq_gaps is not needed.
    upper = math.ceil(x.shape[0] / n) * n
    print("splitting on", n, "with overlap of ", o, "total datapoints:", x.shape[0], "; upper:", upper)
    for i in range(0, upper, n):
        # print(i)
        if i == 0:
            padded = np.zeros((o + n + o, x.shape[1]))  ## pad with 0's on init
            padded[o:, :x.shape[1]] = x[i:i + n + o, :]
            xpart = padded
        else:
            xpart = x[i - o:i + n + o, :]
        if xpart.shape[0] < i:
            padded = np.zeros((o + n + o, xpart.shape[1]))  ## pad with 0's on end of seq
            padded[:xpart.shape[0], :xpart.shape[1]] = xpart
            xpart = padded

        xpart = np.expand_dims(xpart, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)
        try:
            xx = np.vstack((xx, xpart))
        except UnboundLocalError:  ## on init
            xx = xpart
    print("output: ", xx.shape)
    return (xx)


def remove_seq_gaps(x, y):
    # remove parts that are not annotated <- not ideal, but quickest for now.
    window = 150
    c = 0
    cutout = []
    include = []
    print("filterering.")
    print("before shape x,y", x.shape, y.shape)
    for i in range(y.shape[0]):
        c = c + 1
        if c < window:
            include.append(i)
        if sum(y[i, 0:5]) > 0:
            c = 0
        if c >= window:
            # print ('filtering')
            pass
    x, y = x[include, :], y[include, :]
    print(" after shape x,y", x.shape, y.shape)
    return (x, y)


def normalizesignal(x):
    x = st.zscore(x, ddof=0)
    return x


def normalizesignal_array(x):
    for i in range(x.shape[0]):
        x[i] = st.zscore(x[i], axis=0, ddof=0)
    return x


def LoaddDatFiles(datfiles):
    for datfile in datfiles:
        print(datfile)
        if basename(datfile).split(".", 1)[0] in exclude:  # continue exlude files
            continue
        qf = os.path.splitext(datfile)[0] + '.' + annot_type  # set annot filename
        if os.path.isfile(qf):
            # print("yes",qf,datfile)
            x, y = my_get_ecg_data(datfile)
            x, y = remove_seq_gaps(x, y)

            x, y = splitseq(x, split_size, 0), splitseq(y, split_size,
                                                    0)  ## create equal sized numpy arrays of n size and overlap of o

            x = normalizesignal_array(x)
            ## todo; add noise, shuffle leads etc. ?
            try:  ## concat
                xx = np.vstack((xx, x))
                yy = np.vstack((yy, y))
            except NameError:  ## if xx does not exist yet (on init)
                xx = x
                yy = y
    return (xx, yy)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_session(gpu_fraction=0.8):
    # allocate % of gpu memory.
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def getmodel():
    model = Sequential()
    model.add(Dense(32, W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))  # 1000 2
    model.add(Bidirectional(
        LSTM(32, return_sequences=True)))  # , input_shape=(seqlength, features)) ) ### bidirectional ---><---
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', W_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(dimout, activation='softmax'))  # 6
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return (model)

def mymodel():
    model = Sequential()
    model.add(Dense(32, W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))  # 1000 2
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', W_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(dimout, activation='softmax'))  # 6
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return (model)

def convNN():
    model = Sequential()
    model.add(Conv1D(128, kernel_size=6, padding="same", activation='relu', input_shape=(seqlength, features)))  # 1000 2
    model.add(Bidirectional(LSTM(6, return_sequences=True)))
    model.add(Conv1D(64, kernel_size=6, padding="same", activation='relu', input_shape=(seqlength, features)))
    model.add(Bidirectional(LSTM(6, return_sequences=True)))
    model.add(Conv1D(32, kernel_size=6, padding="same", activation='relu'))
    model.add(Dense(dimout, activation='softmax'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return (model)

def cnn():
    model = Sequential()
    model.add(
        Conv1D(128, kernel_size=6, padding="same", activation='relu', input_shape=(seqlength, features)))  # 1000 2
    model.add(Conv1D(64, kernel_size=6, padding="same", activation='relu', input_shape=(seqlength, features)))
    model.add(Conv1D(32, kernel_size=6, padding="same", activation='relu'))
    model.add(Bidirectional(LSTM(6, return_sequences=True)))
    model.add(Dense(dimout, activation='softmax'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return (model)

def just():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(seqlength, features)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(dimout, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return (model)

def unet():
    def get_crop_shape(target, refer):
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2)

    num_class = dimout
    input_shape = (seqlength, features)
    concat_axis = features
    inputs = Input(shape = input_shape)

    conv1 = Conv1D(64, 6, activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv1D(64, 6, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(128, 6, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(128, 6, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(256, 6, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(256, 6, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(512, 6, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(512, 6, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(1024, 6, activation='relu', padding='same')(pool4)
    conv5 = Conv1D(512, 6, activation='relu', padding='same')(conv5)

    up_conv5 = UpSampling1D(size=2)(conv5)
    ch = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping1D(cropping=(ch))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv1D(512, 6, activation='relu', padding='same')(up6)
    conv6 = Conv1D(256, 6, activation='relu', padding='same')(conv6)

    up_conv6 = UpSampling1D(size=2)(conv6)
    ch = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping1D(cropping=(ch))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv1D(256, 6, activation='relu', padding='same')(up7)
    conv7 = Conv1D(128, 6, activation='relu', padding='same')(conv7)

    up_conv7 = UpSampling1D(size=2)(conv7)
    ch = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping1D(cropping=(ch))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv1D(128, 6, activation='relu', padding='same')(up8)
    conv8 = Conv1D(64, 6, activation='relu', padding='same')(conv8)

    up_conv8 = UpSampling1D(size=2)(conv8)
    ch = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping1D(cropping=(ch))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv1D(64, 6, activation='relu', padding='same')(up9)
    conv9 = Conv1D(64, 6, activation='relu', padding='same')(conv9)

    ch = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding1D(padding=((ch[0], ch[1])))(conv9)
    conv10 = Conv1D(num_class, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model

##################################################################
##################################################################
split_size = 1024
qtdbpath = "data\\qt-database-1.0.0\\"  ## first argument = qtdb database from physionet.
perct = 0.9 # percentage training
percv = 0.1 # percentage validation

exclude = set()
"""exclude.update(
    ["sel35", "sel36", "sel37", "sel50", "sel102", "sel104", "sel221", "sel232", "sel310"])  
    # no P annotated:"""

# load data
datfiles = glob.glob(qtdbpath + "*.dat")
xxt, yyt = LoaddDatFiles(datfiles[:round(len(datfiles) * perct)])  # training data.
xxt, yyt = unison_shuffled_copies(xxt, yyt)  ### shuffle
xxv, yyv = LoaddDatFiles(datfiles[-round(len(datfiles) * percv):])  ## validation data.

seqlength = xxt.shape[1]
features = xxt.shape[2]
dimout = yyt.shape[2]
print("xxv/validation shape: {}, Seqlength: {}, Features: {}".format(xxv.shape[0], seqlength, features))

# call keras/tensorflow and build model
KTF.set_session(get_session())
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# ----------------------------------------------
# set info
epochs = 20
model_name = 'unet_real_' + annot_type + "_" + str(epochs) + '.h5'
# ----------------------------------------------
with tf.device('/gpu:0'):  # switch to /cpu:0 to use cpu
    if not os.path.isfile(model_name):
        model = unet()
        history = model.fit(xxt, yyt, validation_data=(xxv, yyv), batch_size=10, epochs=epochs, verbose=1)
        model.save(model_name)
        plot_history(history)
    """
    model = load_model(model_name)
    score, acc = model.evaluate(xxv, yyv, batch_size=4, verbose=1)
    print('Test score: {} , Test accuracy: {}'.format(score, acc))
    """
