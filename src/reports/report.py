import os
import sys

import pandas as pd
import numpy as np

from config.aplication import CONF

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

    def test_model_report(self): # função sobre todos os calculos de testes
        path = CONF["path"] + CONF["name"] + CONF["data"]["path"] + "/" + str(CONF["model"]["LTSM"]["epochs"]) + ".csv"

        df = None
        if (os.path.exists(path)): df = pd.read_csv(path, index_col='Date')

        ax_df = pd.DataFrame(self.pred, columns=CONF["data"]["target"]["description"])
        ax_df= ax_df.T
        ax_df.columns = ["y"]
        ax_df = ax_df.sort_values(by="y", ascending=False)

        self.df['target_pred'] = ax_df.index[0]
        self.df['target_percent_pred'] = ax_df["y"][0]
        self.df["money_return"] = np.zeros(len(self.df))

        if (os.path.exists(path)): self.df = self.df.append(df)

        self.df["target_pred"] = self.df["target_pred"].astype("int32")
        self.df["target"] = self.df["target"].astype("int32")
        self.df["right"] = self.df["target_pred"] == self.df["target"]
        self.df["target_percent_pred"] = pd.to_numeric(self.df["target_percent_pred"])

        self.money_return()

        self.df.to_csv(path)

    def money_return(self) -> None:
        if (self.df["target_pred"][0] == -1):
            self.calc_money_return("High", "Low")
        elif (self.df["target_pred"][0] == 1):
            self.calc_money_return("Low", "High")
        else:
            self.df["money_return"][0] = 0

    def calc_money_return(self, stop: str, goal: str) -> None:
        if (self.df["target"][0] == self.df["target_pred"][0]):
            self.calc_percent_money_return(goal, 1, -1)
        elif (self.df["target"][0] == 0):
            i = 0
            while ((self.df["money_return"][0] == 0) & (i < len(self.df) - 1)):
                i = i + 1
                self.sub_calc_money_return(i, stop, goal)
        else:
            self.calc_percent_money_return(stop, -1, 1)
    
    def sub_calc_money_return(self, i: int, stop: str, goal: str) -> None:
        count = 0
        if ((self.df[goal][i] < self.df["Low"][0]) | (self.df[goal][i] > self.df["High"][0])):
            count = count + 1
            self.calc_percent_money_return(goal, 1, -1)
        if ((self.df[stop][i] < self.df["Low"][0]) | (self.df[stop][i] > self.df["High"][0])):
            count = count + 1
            self.calc_percent_money_return(stop, -1, 1)
        if (count == 2):
            self.df["money_return"][0] = None

    def calc_percent_money_return(self, x: str, a: int, b: int) -> None:
        money_return = (self.df["Close"][0] / self.df[x][0])
        money_return = money_return + 1 if money_return < 0 else money_return - 1
        money_return = money_return * b if money_return < 0 else money_return * a
        self.df["money_return"][0] = money_return

    def print_analisys(self) -> None:
        print("\n")
        print(self.df)
        print("\n")
        
        # filtra os acertos/erros a partir de uma referencia
        self.filter_percent([40, 50 ,60 ,70 ,80 ,90])

    def filter_percent(self, percent: int) -> None:
        for i in percent:
            df = self.df[(self.df["target_percent_pred"] >= (i / 100) )]
            
            print("--->")
            print("\tpercentual " + str(i))
            pred = df["right"]
            print(pred.value_counts())
            print(pred.value_counts(normalize=True))
            print(df.loc[:, ["right", "target_pred", "target"]].groupby(['target', 'target_pred']).count())

            try:
                self.print_money_return(i)
            except:
                print("...")
            print("\n")

    def print_money_return(self, percent: int):

        # parametros da plataforma
        multiplicador = 500  # multiplcador de $
        percentual = 0.12 # percentual minimo de lucro ou prejuizo da plataforma
        percentual_max = 0.99 # percentual minimo de lucro ou prejuizo da plataforma
        investimento = 10 # reais
        taxa = 0.42 # relacioando a -> investimento = 10 
        
        df_money = self.df[self.df["target_percent_pred"] >= (percent / 100)]
        df_percent_range = df_money[((df_money["money_return"] > (percentual / multiplicador)) & (df_money["money_return"] < (percentual_max / multiplicador))) | (((df_money["money_return"] > -(percentual_max / multiplicador))& (df_money["money_return"]  < -(percentual / multiplicador))))]
        trade_in = df_percent_range[df_percent_range["target"] != 0]["target"].count() # numero de trades efetuados

        print("\n\tLucro/Prejuizo real ~= R$ : " + str(round(df_percent_range["money_return"].sum() * multiplicador * investimento - (trade_in * taxa), 2)))
        print("\tLucro/Prejuizo sem taxas ~= R$ : " + str(round((df_percent_range["money_return"].sum() * multiplicador * investimento), 2)))
        print("\tLucro/Prejuizo sem taxas e minimos ~= R$ : " + str(round((df_money["money_return"].sum() * multiplicador * investimento), 2)))

    def pred(self):
        pass

    def print_index(self, index, size):
        print('\n------------------')
        print('index %s - size %s' %(index, size))
        print('------------------\n')
