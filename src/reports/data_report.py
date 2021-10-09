import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Data_report():
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        self.scaler = scaler
        self.config = config
        self.mode = mode

    def set_df_origin(self, x, y) -> None:
        x['target'] = y['target'].values
        self.df_origin = x

    def set_df_test(self, x, y) -> None:
        self.df_x_test = x
        self.df_y_test = y

    def set_df_origin_scaller(self, x, y) -> None:
        self.df_x_origin_scaller = x
        self.df_y_origin_scaller = y