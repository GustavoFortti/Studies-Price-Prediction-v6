import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Pred_report():
    def __init__(self, config: dict, scalar: object, mode: str) -> None:
        self.scalar = scalar
        self.config = config
        self.mode = mode

    def print_resp(self, pred) -> None:
        if (self.config['model']['type'] == 1):
            ax_df = pd.DataFrame(pred, columns=self.config['data']["target"]["description"])
            ax_df = ax_df.T
            ax_df.columns = ["target"]
            out = ax_df.sort_values(by="target", ascending=False)
            print(out)
        else: 
            out = self.scaler.inverse_transform(pred)
            print(out)

        f = open("./notebooks/out.txt", 'a')
        f.write(self.config['name'] + ' - ' + str(out) + ' - ' + str(self.config['data']['target']['columns']) + '\n')