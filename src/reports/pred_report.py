import os
import sys
from datetime import date

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Pred_report():
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        self.scaler = scaler
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
            out = self.scaler['target'].inverse_transform(pred)
            print(out)
        
        data = {"name": self.config['name'], "pred": str(out[0]), "target": self.config['data']['target']['columns'][0]}
        f = self.config['data']['path'] + date.today().strftime("%Y%m%d") + ".csv"
        if (os.path.isfile(f)): df = pd.read_csv(f, names=['name', 'pred', 'target'])
        else: df = pd.DataFrame(data, index=[0])
        if(len(df) > 1): df = df.append(data)

        df.to_csv(f, index=False)