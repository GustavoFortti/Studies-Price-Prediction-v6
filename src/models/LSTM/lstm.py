import sys
from keras.engine.keras_tensor import KerasTensor

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU
from keras import backend as K
import keras
import random
from sklearn.preprocessing import Binarizer
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
        input_tensor = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        self.model.add(LSTM(300, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(300, return_sequences=False))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(loss=self.custon_loss(input_tensor), optimizer='adam', metrics=['mean_absolute_percentage_error'])
        self.model.fit(x_train, y_train, epochs=35, batch_size=42, shuffle=True, validation_data=(x_test, y_test), verbose=1)

        # self.model.summary()
        self.print_graph()
        self.model.save(self.path)
    
    def custon_loss(self, input_tensor):
        def loss(y_actual, y_predicted):
            mse = K.mean(K.sum(K.square(y_actual - y_predicted)))
            ax_input = input_tensor[0][-1:][0][:1]
            return K.switch((K.greater_equal(ax_input, y_predicted) & (K.greater_equal(ax_input, y_actual)) | (K.less_equal(ax_input, y_predicted)) & (K.less_equal(ax_input, y_actual))), mse, (mse * 10))
        return loss

    def predict(self, x) -> np.array:
        self.print_graph()
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)

    def print_graph(self):
        self.report.print_regression_train(self.model, self.report.df_x_test, self.report.df_y_test, self.report.df_x_origin_scaller, self.report.df_y_origin_scaller, self.report.df_origin)