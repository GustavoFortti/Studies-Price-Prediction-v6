import os
import sys
from copy import deepcopy

import pandas as pd
from src.data_manager.indicators_manager import Inticators_manager

from src.services.api import Api_market

class Data_generated():
    def __init__(self, mode: str, config: dict) -> None:
        self.config = config
        self.path = config.path + config.name
        self.mode = mode

        api = Api_market(mode, config)
        df = api.data
        # df = api.no_api()
    
        self.size = len(df)
        self.reduce = int(self.size / config.data['reduce'])
        print(df)
        print(config.data['reduce'])
        print(self.reduce)

        if (((self.mode == 'tr') | (self.mode == 'td') | (self.mode == 'te')) & (os.path.isfile(self.path + '/data.csv'))): self.predictor = self.read_data()
        elif ((self.mode != 'te')): self.predictor = self.generate_data(deepcopy(df), 'predictor')
        self.target = self.generate_data(deepcopy(self.predictor.loc[:, config.data['predict']['columns']]), 'target')
        if (self.mode == 'gd'): sys.exit()
        
    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path + '/data.csv', index_col='Date')

    def generate_data(self, data, _type) -> pd.DataFrame:
        indicators = Inticators_manager(_type, self.config)

        if (_type == 'target'): return indicators.generate(data)
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
        return self.target

    def get_reduce(self) -> int:
        return self.reduce