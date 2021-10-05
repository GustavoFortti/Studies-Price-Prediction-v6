import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, max_error

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
        x, y = df_x_test[:-1, :], df_y_test[:-1, :]
        df_x_origin_scaller, df_y_origin_scaller = df_x_origin_scaller[:-1, :5], df_y_origin_scaller[:-1, :5]
        target = self.config['data']["target"]["columns"][0]

        df = pd.DataFrame(df_x_origin_scaller, columns=self.config['data']["predict"]["columns"])
        df = df.set_index(index)
        df['y'] = df_y_origin_scaller
        df['bool'] = [0 if j >= i else 1 for i, j in zip(df.y, df[target])]
        # pd.set_option("display.max_rows", None, "display.max_columns", None)


        pred = model.predict(x)
        df['pred'] = pred
        df['p_bool'] = [0 if j > i else 1 for i, j in zip(df.pred, df[target])]
        df['right'] = df['bool'] == df['p_bool']
        
        print(df)
        print('\n')

        def adjusted_r2(y_test, y_pred, n_features):
            adj_r2 = (1 - ((1 - r2_score(y_test, y_pred)) * (len(y_test) - 1)) / (len(y_test) - n_features - 1))
            return adj_r2
                
        print("MSE = " + str(mean_squared_error(df['y'], df['pred'])))
        print("RMSE = " + str(mean_squared_error(df['y'], df['pred'], squared=False)))
        print("MAPE = " + str(mean_absolute_percentage_error(df['y'], df['pred']) * 100))
        print("R2 = " + str(r2_score(df['y'], df['pred'])))
        print("adjust R2 = " + str(adjusted_r2(df['y'], df['pred'], 105)))
        print("RMSLE = " + str(mean_squared_log_error(df['y'], df['pred'])))
        print("explained_variance_score = " + str(explained_variance_score(df['y'], df['pred'])))
        print("mean_poisson_deviance = " + str(mean_poisson_deviance(df['y'], df['pred'])))
        print("mean_gamma_deviance = " + str(mean_gamma_deviance(df['y'], df['pred'])))
        print("mean_tweedie_deviance = " + str(mean_tweedie_deviance(df['y'], df['pred'])))
        print("max_error = " + str(max_error(df['y'], df['pred'])))

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

        df.to_csv('./notebooks/pred.csv')