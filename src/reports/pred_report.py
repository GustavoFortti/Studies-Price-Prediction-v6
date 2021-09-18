import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        if((self.config['model']['type'] == 2) & (self.mode == 'te')): self.print_regression_test(pred, df_origin, out)

    def print_regression_test(self, pred, df_origin, out):
        
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

        file = self.config['data']['path'] + "test.csv"
        flag = os.path.isfile(file)
        if (flag): df = pd.read_csv(file)
        else: df = pd.DataFrame(data, index=[0])
        if(flag): df = df.append(data, ignore_index=True)

        print(df)

        df.to_csv(file, index=False)
    
    def print_regression_train(self, model, df_x_test, df_y_test, df_x_origin_scaller, df_y_origin_scaller, df_origin):
        index = df_origin.index[:-1]
        print(df_origin)
        x, y = df_x_test[:-1, :], df_y_test[:-1, :]
        df_x_origin_scaller, df_y_origin_scaller = df_x_origin_scaller[:-1, :1], df_y_origin_scaller[:-1, :1]

        df = pd.DataFrame(df_x_origin_scaller, columns=['x'])
        df['y'] = df_y_origin_scaller
        df['bool'] = [0 if j > i else 1 for i, j in zip(df.y, df.x)]
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        pred = model.predict(x)
        df['pred'] = pred
        df['p_bool'] = [0 if j > i else 1 for i, j in zip(df.pred, df.x)]
        df['right'] = df['bool'] == df['p_bool']
        
        print(df)
        print('\n')
        print(df['right'].value_counts(normalize=True))
        print(df['right'].value_counts())

        print('\n')
        print(df.describe())

        plt.plot(index, y, label = "y")
        plt.plot(index, pred, label = "pred")
        plt.grid(True)
        plt.xticks(rotation=45) 
        
        plt.show()

        df = df.set_index(index)
        df.to_csv('./notebooks/pred.csv')