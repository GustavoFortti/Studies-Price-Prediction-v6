import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Pred_report():
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        self.scaler = scaler
        self.config = config
        self.mode = mode

    def print_resp(self, pred, df_origin) -> None:
        if (self.config['model']['type'] == 1):
            ax_df = pd.DataFrame(pred, columns=self.config['data']["target"]["description"])
            ax_df = ax_df.T
            ax_df.columns = ["target"]
            out = ax_df.sort_values(by="target", ascending=False)
        else: 
            out = self.scaler['target'].inverse_transform(pred)

        print(out)
        if(self.config['model']['type'] == 2): self.print_regression(pred, df_origin, out)

    def print_regression(self, pred, df_origin, out):
        
        origin = self.df_origin[self.config['data']['target']['columns'][0]].values[0]
        target = self.df_origin['target'].values[0]
        pred = np.ndarray.tolist(out[0])
        direction_pred = [1 if i > origin else 0 for i in pred]
        direction_target = 1 if target > origin else 0
        erro = np.ndarray.tolist(abs(pred - target))

        data = {
            "date": df_origin.index[0],
            "name": self.config['name'], 
            "target_name": self.config['data']['target']['columns'][0],
            "origin": origin,
            "pred": str(pred), 
            "target": target,
            "direction_target": direction_target,
            "direction_pred": str(direction_pred),
            "erro": str(erro)
        }

        file = self.config['data']['path'] + df_origin.index[0] + ".csv"

        if (os.path.isfile(file)): df = pd.read_csv(file, names=data.keys, index=[0])
        else: df = pd.DataFrame(data, index=[0])
        if(len(df) > 1): df = df.append(data)

        print(df)

        df.to_csv(file, index=False)