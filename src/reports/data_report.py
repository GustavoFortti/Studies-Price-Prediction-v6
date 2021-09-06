import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Data_report():
    def __init__(self, config: dict, scalar: object, mode: str) -> None:
        super().__init__()

        self.scalar = scalar
        self.config = config
        self.mode = mode

    def set_df_origin(self, x, y) -> None:
        x = x.iloc[:, :4]
        x['target'] = y['target'].values
        self.df = x
        if (self.mode == 'pr'): print(self.df)

    def set_df_end(self, x, y, index) -> None:
        ax_array = x[-(1 + index):-(index)][0][-1:][0]
        ax_array = np.append(ax_array, y[-(1 + index):-(index)][0][0])

        # for i, j in zip(self.df, ax_array):
        #     if (round(self.df[i].values[0], 3) != round(j, 3)): 
        #         print("Error target: unaligned data")
        #         sys.exit()


