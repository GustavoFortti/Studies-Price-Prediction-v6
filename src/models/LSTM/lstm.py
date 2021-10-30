from random import randint
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.layers import LeakyReLU
from tensorflow.keras.losses import Loss
from keras import backend as K
import keras

from src.models.LSTM.genetic_algorithm import genetict_algorithm


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
        gene_structure = {
            "dropout": {
                "range": (0.20, 0.50),
                "size": 3
            },
            "LSTMneurons": {
                "range": (10, 100),
                "size": 3
            },
            "DENSEneurons": {
                "range": (10, 100),
                "size": 3
            },
            "activation": {
                "range": ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential'],
                "size": 3
            }
        }

        size = 20

        ga = genetict_algorithm(gene_structure, size)
        max_score = 0.0
        i = 0
        while (max_score < 0.9):
            print(f'\generation {str(i)}')
            score = []
            j = 0
            for k in ga.population:
                print(f'\ntrain {str(j)}')
                print(k)
                f = open('model.txt', 'a')
                f.write(str(k))
                score.append(self.LSTM(data, report, k))
                f.write(str(score))
                f.close()
                print("\nscore")
                print(score)
                j = j + 1
            ga.evolution(score)
            max_score = np.amax(score)
            print(max_score)
            i = i + 1
        
    def LSTM(self, data: object, report: object, params: list):
        self.data = data
        self.report = report
        x_train, x_test, y_train, y_test = data.get_train_test()

        self.model = Sequential()
        self.model.add(LSTM(params[3]['LSTMneurons_0'], return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(params[0]['dropout_0']))

        self.model.add(LSTM(params[4]['LSTMneurons_1'], return_sequences=True))
        self.model.add(Dropout(params[1]['dropout_1']))

        self.model.add(LSTM(params[5]['LSTMneurons_2'], return_sequences=False))
        self.model.add(Dropout(params[2]['dropout_2']))

        self.model.add(Dense(params[6]['DENSEneurons_0'], activation='tanh'))
        self.model.add(Dense(params[7]['DENSEneurons_1'], activation='tanh'))
        self.model.add(Dense(params[8]['DENSEneurons_2'], activation='tanh'))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
        self.model.fit(x_train, y_train, epochs=3, batch_size=42, shuffle=True, validation_data=(x_test, y_test), verbose=1)

        return self.print_graph() # score
        # self.model.save(self.path)

    def regression(self, data: object, report: object) -> None:
        self.data = data
        self.report = report
        x_train, x_test, y_train, y_test = data.get_train_test()

        self.model = Sequential()

        self.model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(0.20))

        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.20))

        self.model.add(Dense(50, activation='tanh'))
        self.model.add(Dense(50, activation='tanh'))
        self.model.add(Dense(50, activation='tanh'))
        self.model.add(Dense(50, activation='tanh'))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
        self.model.fit(x_train, y_train, epochs=10, batch_size=42, shuffle=True, validation_data=(x_test, y_test), verbose=1)

        score = self.print_graph()
        print(score)
        self.model.save(self.path)

    def predict(self, x) -> np.array:
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)

    def print_graph(self):
        return self.report.print_resp(self.model.predict(self.report.df_x_test), self.report.df_origin)