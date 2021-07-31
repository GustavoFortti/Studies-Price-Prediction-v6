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

        if ((self.mode == 'pr') | (self.mode == 'gd')): self.read_data()
        else: 
            df = Api_market().data
            self.size = len(df)
            self.reduce = int(self.size / CONF['data']['reduce'])

            self.predictor = self.generate_data(deepcopy(df), 'predictor')
            self.target = self.generate_data(deepcopy(df), 'target')
            if (self.mode == 'gd'): sys.exit()
        
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path + '/data.csv')

    def generate_data(self, data, type) -> pd.DataFrame:
        for i in range(self.size, 0, -self.reduce): # cria partições para o dataframe
            if (self.reduce > i): break
            ax_df = data.iloc[(i - self.reduce):i, :]

            # ax_df = Inticators_manager(deepcopy(ax_df), type).data

            if (i == self.size): df = ax_df.iloc[::-1] # inverti as partições e depois uni elas
            else: df = df.append(ax_df.iloc[::-1])

        if (type == 'predictor'): df.to_csv(self.path + '/data.csv')
        return df.iloc[::-1]

    def get_predictor(self) -> pd.DataFrame:
        return self.predictor

    def get_target(self) -> pd.DataFrame:
        return self.target