from copy import deepcopy
import sys

import pandas as pd
import numpy as np

from src.data_manager.data_generated import Data_generated

import tensorflow as tf
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split


class Data_manager():
    def __init__(self, mode: str, index: int, data_report: object, config: dict, scaler: object) -> None:
        self.scaler = scaler
        self.mode = mode
        self.config = config

        data_gen = Data_generated(mode, config)
        x, y = data_gen.get_predictor(), data_gen.get_target()
        
        data_report.set_df_origin(x, y)

        x, y = self.pre_shape_data(x, y, config['data']['timesteps'], data_gen.get_reduce()) # novo shape para o dataframe - 3 dimensões
        size = int(len(x) * config['model']['slice'])
        
        self.x, self.y = self.df_slice(mode, index, x, size), self.df_slice(mode, index, y, size)
        if (mode == 'tr'): self.adjust_data(self.x, self.y, config['data']['target']['description'])
        
        data_report.set_df_end(x, y, index)

    def df_slice(self, mode: str, index: int, df: pd.DataFrame = None, size: int = 0) -> pd.DataFrame:
        if (size < index): sys.exit()
        return df[:-size] if 'tr' == mode else (df[-(1 + index):-(index)] if 'te' == mode else df[-1:]) #  df[-1:] - 'pr' == mode

    def pre_shape_data(self, x: DataFrame, y: np.array, timesteps: int, reduce: int) -> list:
        x_temp, y_temp = [], []
        init = 100
        
        for i in range(0, len(x), reduce):
            x_aux, y_aux = self.shape_data(x.iloc[i + init:(i + reduce), :], y[i + init:(i + reduce)], timesteps)
            if (self.mode != 'pr'):
                x_aux, y_aux = x_aux[:-1], y_aux[:-1]
            if (x_temp == []):
                x_temp, y_temp = x_aux, y_aux
            else:
                x_temp, y_temp = np.concatenate((x_temp, x_aux)), np.concatenate((y_temp, y_aux))

        return [np.array(x_temp), np.array(y_temp)]

    def shape_data(self, x: DataFrame, y: np.array, timesteps: int) -> list:
        x = self.scaler.fit_transform(x)
        if (self.config['model']['type'] == 2): y = self.scaler.fit_transform(y)

        reshaped = []
        for i in range(timesteps, x.shape[0] + 1):
            reshaped.append(x[i - timesteps:i])

        return [np.array(reshaped), np.array(y[timesteps-1:])]

    def adjust_data(self, x: np.array, y: np.array, categorical: dict, split: float=0.3) -> None:
        self.x_train, self.x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        if (self.config['model']['type'] == 1): self.y_train, self.y_test = tf.keras.utils.to_categorical(y_train, len(categorical)), tf.keras.utils.to_categorical(y_test, len(categorical)) 
        else: self.y_train, self.y_test = y_train, y_test

    def get_train_test(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_x(self) -> np.array:
        return self.x

    def get_y(self) -> np.array:
        return self.y