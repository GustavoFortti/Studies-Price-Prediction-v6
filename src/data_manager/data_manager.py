import sys

import pandas as pd
import numpy as np

from config.aplication import CONF
from src.data_manager.data_generated import Data_generated

from keras.utils import to_categorical
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Data_manager():
    def __init__(self, mode: str, index: int=0) -> None:
        
        data_gen = Data_generated(mode)
        x = data_gen.get_predictor()
        y = data_gen.get_target()

        print(x)
        print(y)

        x, y = self.pre_shape_data(x, y, CONF['data']['timesteps'], data_gen.get_reduce()) # divide o dataframe em bloco de 3d

        if (mode == 'tr'):
            size = int(len(x) * CONF['model']['slice'])
            x = x[:-size]
            y = y[:-size]
            self.adjust_data(x, y, CONF['data']['target']['categorical'])
        if (mode == 'te'):
            self.x = x[-(2 + index):-(1 + index)] # pega apenas 1 bloco para fazer a predição teste
            self.y = y[-(2 + index):-(1 + index)]
        if (mode == 'pr'):
            self.x = x[-1:] # predição - pega apenas o ultimo bloco

        if (mode == 'td'): 
            # reports
            sys.exit()

    def pre_shape_data(self, x: DataFrame, y: np.array, timesteps: int, reduce: int) -> list:
        x_temp = []
        y_temp = []
        init = 31
        for i in range(0, len(x), reduce):
            x_aux, y_aux = self.shape_data(x.iloc[i + init:(i + reduce), :], y[i + init:(i + reduce)], timesteps)
            for i, j in zip(x_aux, y_aux): 
                x_temp.append(i) 
                y_temp.append(j) 

        return [np.array(x_temp), np.array(y_temp)]

    def shape_data(self, x: DataFrame, y: np.array, timesteps: int, init: int=31) -> list:
        scaler = StandardScaler() 
        x = scaler.fit_transform(x)

        reshaped = []
        for i in range(timesteps, x.shape[0] + 1):
            reshaped.append(x[i - timesteps:i])

        x = np.array(reshaped)
        y = np.array(y[timesteps-1:])

        return [x, y]

    def adjust_data(self, x: np.array, y: np.array, categorical: dict, split: float=0.3) -> None:
        self.x_train, self.x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        self.y_train, self.y_test = to_categorical(y_train, categorical), to_categorical(y_test, categorical) 
    
    def get_train_test(self):
        return self.x_train, self.x_test, self.y_train, self.y_test