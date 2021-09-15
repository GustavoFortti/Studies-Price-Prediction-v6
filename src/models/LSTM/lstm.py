import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU
from keras import backend as K
import matplotlib.pyplot as plt
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
        x_train, x_test, y_train, y_test = data.get_train_test()
        x, y = data.get_x_pred_vector()[1:, :], data.get_y_pred_vector()[1:, :]
        
        df = pd.DataFrame(report.df_test_scaler[:-1, :1], columns=['x'])
        df['y'] = y
        df['bool'] = [1 if i > j else 0 for i, j in zip(df.y, df.x)]

        self.model = Sequential()

        self.model.add(LSTM(242, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(122, return_sequences=False))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(32, activation='softmax'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
        self.model.fit(x_train, y_train, epochs=20, batch_size=42, shuffle=True, validation_data=(x_test, y_test), verbose=1)
        # self.model.summary()

        pd.set_option("display.max_rows", None, "display.max_columns", None)
        r = report.df_test.index[1:]
        pred = self.model.predict(x)
        df['pred'] = pred
        df['p_bool'] = [1 if i > j else 0 for i, j in zip(df.y, df.pred)]
        print(df)
        plt.plot(r, y, label = "y")
        plt.plot(r, pred, label = "pred")
        plt.grid(True)
        
        plt.show()


    def save(self) -> None:
        self.model.save(self.path)
    
    def predict(self, x) -> np.array:
        model = tf.keras.models.load_model(self.path)
        return model.predict(x)