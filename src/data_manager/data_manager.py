from copy import deepcopy
import sys

import pandas as pd
import numpy as np

from src.data_manager.data_generated import Data_generated

from keras.utils import to_categorical
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split


class Data_manager():
    def __init__(self, mode: str, index: int, report: object, config: dict, scaler: object) -> None:
        self.scaler = scaler
        self.mode = mode
        self.config = config

        data_gen = Data_generated(mode, config)
        x = data_gen.get_predictor()
        y = data_gen.get_target()
        
        if (mode == 'pr'):
            print(x[-1:])
            print(y[-1:])
        # if (mode != 'pr'): report.set_df_origin(x[-(1 + index):-(index)], y[-(1 + index):-(index)])
        # else: report.set_df_origin(x[-1:], y[-1:])

        x, y = self.pre_shape_data(x, y, config.data['timesteps'], data_gen.get_reduce()) # divide o dataframe em bloco de 3d

        size = int(len(x) * config.model['slice'])
        if (mode == 'tr'):
            x = x[:-size]
            y = y[:-size]
            self.adjust_data(x, y, config.data['target']['categorical'])
        if (mode == 'te'):
            if (size > index):
                report.print_index(index, size)
                self.x = x[-(1 + index):-(index)] # pega apenas 1 bloco para fazer a predição teste
                self.y = y[-(1 + index):-(index)]
        if (mode == 'pr'):
            self.x = x[-1:] # predição - pega apenas o ultimo bloco
        
        if (mode == 'td'): 
            print(x)
            print(y)
            # report.set_df_end(x, y, index)
            # report.set_df_end_target(y, index)
            sys.exit()

    def pre_shape_data(self, x: DataFrame, y: np.array, timesteps: int, reduce: int) -> list:
        x_temp = []
        y_temp = []
        init = 31
        for i in range(0, len(x), reduce):
            x_aux, y_aux = self.shape_data(x.iloc[i + init:(i + reduce), :], y[i + init:(i + reduce)], timesteps)
            if (self.mode != 'pr'):
                x_aux = x_aux[:-1]
                y_aux = y_aux[:-1]

            if (x_temp == []):
                x_temp = x_aux
                y_temp = y_aux
            else:
                x_temp = np.concatenate((x_temp, x_aux))
                y_temp = np.concatenate((y_temp, y_aux))

        return [np.array(x_temp), np.array(y_temp)]

    def shape_data(self, x: DataFrame, y: np.array, timesteps: int) -> list:
        x = self.scaler.fit_transform(x)
        if (self.config.model['type'] == 2): y = self.scaler.fit_transform(y)

        reshaped = []
        for i in range(timesteps, x.shape[0] + 1):
            reshaped.append(x[i - timesteps:i])

        x = np.array(reshaped)
        y = np.array(y[timesteps-1:])

        return [x, y]

    def adjust_data(self, x: np.array, y: np.array, categorical: dict, split: float=0.3) -> None:
        self.x_train, self.x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        if (self.config.model['type'] == 1): self.y_train, self.y_test = to_categorical(y_train, categorical), to_categorical(y_test, categorical) 
        else: self.y_train, self.y_test = y_train, y_test

    def get_train_test(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_x(self) -> np.array:
        return self.x

    def get_y(self) -> np.array:
        return self.y