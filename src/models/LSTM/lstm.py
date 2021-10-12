from random import randint
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Input
from tensorflow.keras.losses import Loss
from keras import backend as K
import keras
class LTSM_model():
    def __init__(self, config: dict) -> None:
        self.epochs = config['model']['LSTM']['epochs']
        self.name = 'M1'
        self.path = config['path'] + config['name'] + '/models/epochs_' + str(self.epochs) + '_' + self.name +  '_lstm_model'
        self.categorical = len(config['data']['target']['description'])

    def classification(self, x_train, x_test, y_train, y_test, x, y) -> None:
        self.model = Sequential()

        self.model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.categorical, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=28, shuffle=True, validation_data=(x_test, y_test), verbose=1)

    def regression(self, data: object, report: object) -> None:
        self.data = data
        self.report = report
        x_train, x_test, y_train, y_test = data.get_train_test()

        self.model = Sequential()
        self.model.add(LSTM(300, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(300, return_sequences=False))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=42, shuffle=True, validation_data=(x_test, y_test), verbose=1)


        # self.model.summary()
        self.print_graph()
        self.model.save(self.path)

    # def regression_2(self, data: object, report: object) -> None:
    #     self.data = data
    #     self.report = report
    #     x_train, x_test, y_train, y_test = data.get_train_test()

    #     input_tensor = tf.constant(x_train, dtype='float32', shape=(x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    #     index = np.array(range(x_train.shape[0]))
    #     _input = Input(shape=(x_train.shape[1], x_train.shape[2]), dtype='float32')
    #     _target = Input(shape=(None, y_train.shape[1]), dtype='float32')
    #     _out = Input(shape=(None, y_train.shape[1]), dtype='float32')
    #     seq = Sequential()(_input)

    #     l1 = LSTM(300, dropout=0.25, return_sequences=True)(seq)
    #     print(l1)
    #     l2 = LSTM(300, dropout=0.25, return_sequences=False)(l1)
    #     print(l2)

    #     d1 = Dense(50, activation='relu')(l2)
    #     print(d1)
    #     d2 = Dense(30, activation='relu')(d1)
    #     print(d2)
    #     d3 = Dense(16, activation='relu')(d2)
    #     print(d3)
    #     out = Dense(1)(d3)
    #     model = Model(inputs=[_target, _input], outputs=out)
    #     print(model)

    #     model.add_loss(self.loss(_target, _input, _out))
    #     model.compile(loss=None, optimizer='adam')
    #     model.fit(x=[x_train, x_train], y=y_train, epochs=3, batch_size=42, shuffle=True)#, validation_data=(x_test, y_test), verbose=1)

    #     # self.model.summary()
    #     # self.print_graph()
    #     # self.model.save(self.path)
    
    # def loss(self, y_actual, y_predicted, _input):
    #     print(y_predicted)
    #     print(y_actual)
    #     print(_input)
    #     sys.exit()
    #     mse = K.mean(K.sum(K.square(y_actual - y_predicted)))
    #     mse = tf.reshape(mse, [1, 1])
    #     return lambda: mse
    #     y_actual = keras.layers.core.Reshape([1, 1])(y_actual)[0]
    #     # ax_input = tf.reshape(input_tensor[0][-1:][0][:1], [1, 1])
    #     # ax_input = tf.keras.backend.variable(ax_input)
    #     # print(input_tensor)

    #     greater_equal = tf.reshape(tf.math.logical_and(tf.math.greater_equal(tf.constant(1.1), y_actual), tf.math.greater_equal(tf.constant(1.1), y_predicted))[0], [1, 1])
    #     less_equal = tf.reshape(tf.math.logical_and(tf.math.less_equal(tf.constant(1.1), y_actual), tf.math.less_equal(tf.constant(1.1), y_predicted))[0], [1, 1])
    #     logical_or = tf.reshape(tf.math.logical_or(greater_equal, less_equal)[0], [1, 1])
    #     print(greater_equal)
    #     print(less_equal)
    #     print(logical_or)
    #     tf.cond(tf.equal(greater_equal, tf.constant(True)), lambda: mse, lambda: tf.math.multiply(mse, 10))
    #     return tf.cond(logical_or, lambda: mse, lambda: tf.math.multiply(mse, 10))

    def predict(self, x) -> np.array:
        # self.print_graph()
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)

    def print_graph(self):
        self.report.print_regression_train(self.model, self.report.df_x_test, self.report.df_y_test, self.report.df_x_origin_scaller, self.report.df_y_origin_scaller, self.report.df_origin)

