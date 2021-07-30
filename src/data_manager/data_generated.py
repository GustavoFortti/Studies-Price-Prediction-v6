import sys
from copy import deepcopy

import pandas as pd
from src.data_manager.indicators_manager import Inticators_manager

from src.services.api import Api_market
from config.aplication import CONF

class Data_generated():
    def __init__(self, mode: str) -> None:
        self.mode = mode
        df = Api_market(CONF['market']).data

        self.size = len(df)
        self.reduce = self.size / CONF['data']['reduce']

        if ((self.mode == 'pr') | (self.mode == 'gd')): self.read_data()
        else: 
            self.predictor = self.generate_data(deepcopy(df, 'predictor'))
            self.target = self.generate_data(deepcopy(df, 'target'))
        
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(CONF['path'] + CONF['name'] + '/data.csv')

    def generate_data(self, ax_df, type):
        for i in range(self.size, 0, -self.reduce): # cria partições para o dataframe
            ax_df = ax_df.iloc[(i-self.reduce):i, :]

            ax_df = Inticators_manager(deepcopy(ax_df), type).data

            if (i == self.size): df = ax_df.iloc[::-1] # inverti as partições e depois uni elas
            else: df = df.append(ax_df.iloc[::-1])

        df.to_csv(CONF['path'] + CONF['name'] + '/data.csv', index=False)
        if (self.mode == 'gd'): sys.exit()

    def get_predictor(self) -> pd.DataFrame:
        return self.predictor

    def get_target(self) -> pd.DataFrame:
        return self.target