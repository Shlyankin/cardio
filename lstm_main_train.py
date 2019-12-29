import tensorflow as tf
from keras.models import Model
from cardio.batchflow.batchflow.models.keras.keras_model import KerasModel

from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional  # could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF

class MyLSTM(KerasModel):

    def build(self, *args, **kwargs):
        seqlength = 0
        features = 0
        dimout = 0
        model = Sequential()
        model.add(Dense(32, W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
        model.add(Bidirectional(
            LSTM(32, return_sequences=True)))  # , input_shape=(seqlength, features)) ) ### bidirectional ---><---
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu', W_regularizer=regularizers.l2(l=0.01)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(dimout, activation='softmax'))
        adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print(model.summary())
        return (model)