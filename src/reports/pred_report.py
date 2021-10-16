import os
import sys
from datetime import date, datetime, timedelta

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

    def print_resp(self, pred, df_origin, df_y_test=None) -> None:
        if (self.config['model']['model_type'] == 1):
            ax_df = pd.DataFrame(pred, columns=self.config['data']["target"]["description"])
            ax_df = ax_df.T
            ax_df.columns = ["target"]
            out = ax_df.sort_values(by="target", ascending=False)
        else: 
            out = self.scaler['target'].inverse_transform(pred)

        print(out)
        print(df_origin)
        if (self.mode == 'pr'): print(out)
        if ((self.config['model']['model_type'] == 2) & (self.mode == 'tr')): self.plot_regression(df_origin, out, df_y_test)

    def plot_regression(self, df_origin, out, df_y_test):
        df_y_test = self.scaler['target'].inverse_transform(df_y_test)
        print(df_y_test)
        df = df_origin
        index = df.index.astype('datetime64[ns]')
        
        target = self.config['data']["target"]["columns"][0]
        columns = self.config['data']["predict"]["columns"]
        if (target not in self.config['data']["predict"]["columns"]): columns.append(target)
        df = df.loc[:, columns] 
        df['target'] = df_y_test
        df['pred'] = out
        df['error'] = abs(df['pred'] - df['target'])
        df['bool'] = [0 if j >= i else 1 for i, j in zip(df.target, df[target])]
        df['p_bool'] = [0 if j > i else 1 for i, j in zip(df.pred, df[target])]
        df['right'] = df['bool'] == df['p_bool']

        add_constant_column = lambda col: [col for i in range(len(df))]
        df['name'] = add_constant_column(self.config['name'])
        df['target_name'] = add_constant_column(self.config['data']['target']['columns'][0])
        days_ahead = int(self.config['name'][-1:]) + 1
        df['days_ahead'] = add_constant_column(str(days_ahead))
        slice_i = int(self.config['name'][-1:]) + 1
        df['day_predict'] = np.append([str(i)[:10] for i in index[slice_i:]], [str(timedelta(days=slice_i) + i)[:10] for i in index[-slice_i:]])

        self.graph_analysis(index, df)
        for i in range(1, (len(df) + 1), 25):
            print("====================================================================")
            print("index = " + str(i) + "\n")
            self.graph_analysis(index[i: i+25], df.iloc[i: i+25, :])

        df = df.reset_index()

        file = self.config['data']['path'] + "data_test.csv"
        flag = os.path.isfile(file)
        if (flag): 
            df_ax = pd.read_csv(file)
            df = df_ax.append(df, ignore_index=True)
        df.to_csv(file, index=False)

    def graph_analysis(self, index, df):
        def adjusted_r2(y_test, y_pred, n_features):
            adj_r2 = (1 - ((1 - r2_score(y_test, y_pred)) * (len(y_test) - 1)) / (len(y_test) - n_features - 1))
            return adj_r2
        
        try:
            print("MSE = " + str(mean_squared_error(df['target'], df['pred'])))
            print("RMSE = " + str(mean_squared_error(df['target'], df['pred'], squared=False)))
            print("MAPE = " + str(mean_absolute_percentage_error(df['target'], df['pred']) * 100))
            print("R2 = " + str(r2_score(df['target'], df['pred'])))
            print("adjust R2 = " + str(adjusted_r2(df['target'], df['pred'], 105)))
            print("RMSLE = " + str(mean_squared_log_error(df['target'], df['pred'])))
            print("max_error = " + str(max_error(df['target'], df['pred'])))
            print("explained_variance_score = " + str(explained_variance_score(df['target'], df['pred'])))
            print("mean_poisson_deviance = " + str(mean_poisson_deviance(df['target'], df['pred'])))
            print("mean_gamma_deviance = " + str(mean_gamma_deviance(df['target'], df['pred'])))
            print("mean_tweedie_deviance = " + str(mean_tweedie_deviance(df['target'], df['pred'])))
        except:
            print("calc error")

        print('\n')
        print(df['right'].value_counts(normalize=True))
        print(df['right'].value_counts())

        print('\n')
        print(df.describe())

        plt.plot(index, df.target, label = "target")
        plt.plot(index, df.pred, label = "pred")
        plt.grid(True)
        plt.xticks(rotation=45) 
        plt.show()

        plt.scatter(index, df['error'], label = "error")
        plt.grid(True)
        plt.xticks(rotation=45) 
        plt.show()

    def plot_market_price(self, df):

        def plot_graph_market():
            for idx, val in df.iterrows():
                plt.plot([idx, idx], [val['Low'], val['High']], color='Black')

        def plot_graph_real(col, color):
            plt.plot(df.index, df[col], color=color)
            plt.scatter(df.index, df[col], color=color)

        def plot_graph_pred(days_ahead, df_temp):
            x, y = [], []
            target = []
            color = []
            ax_df = df_temp[df_temp['days_ahead'] == days_ahead]
            [y.append(ax_df['day_predict'].values[0][:10]) for i in range(3)]
            for i, j in zip(colors_0, range(3)):
                target.append(ax_df['target'].values[j])
                x.append(ax_df['pred'].values[j])
                color.append(i)
            plt.scatter(y, x, linewidths=3, s=300, facecolors='none', edgecolors=color)
            
        targets_name = df['target_name'].drop_duplicates().values
        colors_0 = ['red', 'royalblue', 'limegreen']
        colors_1 = ['indianred', 'blue', 'green']

        df = df.sort_values(by='target_name', ascending=False)
        df['days_ahead'] = df['days_ahead'].astype('int32')
        plot_graph_market()
        plot_graph_real('Close', 'Black')
        [plot_graph_real(i, j) for i, j in zip(targets_name, colors_0)]
        [plot_graph_pred((i + 1), df) for i in range(4)]

        plt.xticks(rotation=45) 
        plt.grid(True)
        plt.show()