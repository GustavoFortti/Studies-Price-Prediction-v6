import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data_manager.data_generated import Data_generated

class Data_manager():
    def __init__(self, mode: str, index: int, report: object, config: dict, scaler: object) -> None:
        self.scaler = scaler
        self.mode = mode
        self.config = config
        self.report = report
        self.index = index

        data_gen = Data_generated(self.mode, self.config)
        x, y = data_gen.get_predictor(), data_gen.get_target()
        print(x)
        self.size = int(len(x) * self.config['model']['slice'])

        self.report.set_df_origin(self.df_slice(x, True), self.df_slice(y, True))
        x, y = self.pre_shape_data(x, y, self.config['data']['timesteps'], data_gen.get_reduce()) # novo shape para o dataframe - 3 dimensões

        self.report.set_df_test(self.df_slice(x, True), self.df_slice(y, True))
        self.x, self.y = self.df_slice(x), self.df_slice(y)
        if (self.mode == 'tr'): self.adjust_data()

    def df_slice(self, df: pd.DataFrame = None, is_reverse_tr: bool = False) -> pd.DataFrame:
        if (self.size < self.index): 
            print("Error: index > size")
            sys.exit()
        
        mode = 'rt' if ((self.mode == 'tr') & is_reverse_tr) else self.mode
        
        switch = {
            'tr': df[:-self.size],
            'rt': df[-self.size:],
            'te': df[-(1 + self.index):-(self.index)],
            'pr': df[-1:]
        }

        return switch.get(mode)

    def pre_shape_data(self, x: DataFrame, y: np.array, timesteps: int, reduce: int) -> list:
        x_temp, y_temp = [], []
        init = 100
        
        for i in range(0, len(x), reduce):
            x_aux, y_aux = self.shape_data(x.iloc[i + init:(i + reduce), :], y[i + init:(i + reduce)], timesteps)
            if (self.mode == 'tr'):
                x_aux, y_aux = x_aux[:-1], y_aux[:-1]
            if (x_temp == []):
                x_temp, y_temp = x_aux, y_aux
            else:
                x_temp, y_temp = np.concatenate((x_temp, x_aux)), np.concatenate((y_temp, y_aux))

        return [np.array(x_temp), np.array(y_temp)]

    def shape_data(self, x: DataFrame, y: np.array, timesteps: int) -> list:
        x = self.scaler['predictor'].fit_transform(x)
        if (self.config['model']['type'] == 2): y = self.scaler['target'].fit_transform(y)
        self.report.set_df_origin_scaller(self.df_slice(x, True), self.df_slice(y, True))

        reshaped = []
        for i in range(timesteps, x.shape[0] + 1):
            reshaped.append(x[i - timesteps:i])

        return [np.array(reshaped), np.array(y[timesteps-1:])]

    def adjust_data(self, split: float=0.2) -> None:
        categorical = self.config['data']['target']['description']
        self.x_train, self.x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=split, random_state=46)
        if (self.config['model']['type'] == 1): self.y_train, self.y_test = tf.keras.utils.to_categorical(y_train, len(categorical)), tf.keras.utils.to_categorical(y_test, len(categorical)) 
        else: self.y_train, self.y_test = y_train, y_test

    def get_train_test(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def get_x(self) -> np.array:
        return self.x

    def get_y(self) -> np.array:
        return self.y