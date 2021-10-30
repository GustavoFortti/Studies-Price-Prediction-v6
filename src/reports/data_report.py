import os
import sys

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

pd.options.mode.chained_assignment = None 

class Data_report():
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        self.scaler = scaler
        self.config = config
        self.mode = mode

    def set_df_origin(self, x: DataFrame, y: DataFrame) -> None:
        x['target'] = y['target'].values
        self.df_origin = x
        print(x.loc[:, [self.config['data']["target"]["columns"][0], 'target']])

    def set_df_test(self, x: np.array, y: np.array) -> None:
        self.df_x_test = x
        self.df_y_test = y

    def set_df_origin_scaller(self, x: np.array, y: np.array) -> None:
        self.df_x_origin_scaller = x
        self.df_y_origin_scaller = y