from config.aplication import CONF
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Report():
    def __init__(self) -> None:
        pass

    def set_pred(self, pred) -> None:
        self.pred = pred

    def set_df_origin(self, x, y):
        x = x.iloc[:, :4]
        x['target'] = y['target']
        self.df = x

    def set_df_end(self, x, y, index):
        ax_array = x[-(1 + index):-(index)][0][-1:][0]
        ax_array = np.append(ax_array, y[-(1 + index):-(index)][0][0])

        for i, j in zip(self.df, ax_array):
            if (round(self.df[i].values[0], 3) != round(j, 3)): 
                print("Error target: unaligned data")
                sys.exit()

    def set_df_end_target(self, y, index):
        if (self.df["target"].values != y[-(1 + index):-(index)][0][0]): 
            print("Error target: unaligned data")
            sys.exit()

    def compare_train(): # função de comparação do treinamento com outros trinos
        pass

    def save_test(self): # função sobre todos os calculos de testes
        ax_df = pd.DataFrame(self.pred, columns=CONF["data"]["target"]["description"])
        ax_df= ax_df.T
        ax_df.columns = ["y"]
        ax_df = ax_df.sort_values(by="y", ascending=False)

        self.df['target_pred'] = ax_df.index[0]
        self.df['target_percent_pred'] = ax_df["y"][0]

        # processo

        self.df.to_csv(CONF["path"] + CONF["name"] + CONF["data"]["path"] + "/" + CONF["model"]["LTSM"]["epochs"] + ".csv")

    def pred():
        pass