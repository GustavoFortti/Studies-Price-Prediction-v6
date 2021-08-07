import os
import sys
from copy import deepcopy

import pandas as pd
from src.data_manager.indicators_manager import Inticators_manager

from src.services.api import Api_market
from config.aplication import CONF

class Data_generated():
    def __init__(self, mode: str) -> None:
        self.path = CONF['path'] + CONF['name']
        self.mode = mode

        df = Api_market().data
        self.size = len(df)
        self.reduce = int(self.size / CONF['data']['reduce'])

        if (((self.mode == 'tr') | (self.mode == 'td')) & (os.path.isfile(self.path + '/data.csv'))): self.predictor = self.read_data()
        else: self.predictor = self.generate_data(deepcopy(df), 'predictor')
        self.target = self.generate_data(deepcopy(df), 'target')
        if (self.mode == 'gd'): sys.exit()
        
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path + '/data.csv', index_col='Date')

    def generate_data(self, data, _type) -> pd.DataFrame:
        indicators = Inticators_manager(_type)

        for i in range(self.size, 0, -self.reduce): # cria partições para o dataframe
            if (self.reduce > i): break
            ax_df = data.iloc[(i - self.reduce):i, :]

            ax_df = indicators.generate(ax_df)

            if (i == self.size): df = ax_df.iloc[::-1] # inverti as partições e depois uni elas
            else: df = df.append(ax_df.iloc[::-1])
            if ((self.mode == "pr") | (self.mode == "td")): break

        df = df.iloc[::-1]
        if ((_type == 'predictor') & (self.mode in ['gd', 'tr'])): df.to_csv(self.path + '/data.csv')
        return df

    def get_predictor(self) -> pd.DataFrame:
        return self.predictor

    def get_target(self) -> pd.DataFrame:
        return self.target[1:]

    def get_reduce(self) -> int:
        return self.reduce