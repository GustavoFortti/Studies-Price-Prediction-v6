import os
import sys

import pandas as pd
import numpy as np

from config.aplication import CONF

pd.options.mode.chained_assignment = None 

class Report():
    def __init__(self, config: dict) -> None:
        self.config = config
        self.path = config.path + config.name + config.data["path"] + "/" + str(config.model["LTSM"]["epochs"])

    def pred_model_report(self):
        path = self.path + "_pred.csv"

        df = None
        if (os.path.exists(path)): df = pd.read_csv(path, index_col='Date')
        if (os.path.exists(path)): self.df = self.df.append(df)
        self.df = self.df.drop(columns=['target'])
        self.df['pred'] = self.pred.index[0]
        self.df['pred_percent'] = self.pred.values[0]
        print(self.df)

        # self.df.to_csv(path)

    def test_model_report(self): # função sobre todos os calculos de testes
        path = self.path + "_test.csv"

        df = None
        if (os.path.exists(path)): df = pd.read_csv(path, index_col='Date')

        self.gen_target()   
        
        self.df["money_return"] = np.zeros(len(self.df))

        if (os.path.exists(path)): self.df = self.df.append(df)

        self.df["right"] = self.df["target_pred"] == self.df["target"]

        print(self.df)
        print(self.df['right'].value_counts())
        print(self.df['right'].value_counts(normalize=True))
        print(self.df.loc[:, ["right", "target_pred", "target"]].groupby(['target', 'target_pred']).count())
        print('\n\n\n\n')
        # self.money_return()

        self.df.to_csv(path)

    def set_pred(self, pred) -> None:
        ax_df = pd.DataFrame(pred, columns=self.config.data["target"]["description"])
        ax_df= ax_df.T
        ax_df.columns = ["target"]
        ax_df = ax_df.sort_values(by="target", ascending=False)
        self.pred = ax_df

    def set_df_origin(self, x, y):
        print(y)
        print(x)
        x = x.iloc[:, :4]
        x['target'] = y['target'].values
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

    def gen_target(self):
        print(self.df["target"])
        self.df["target"] = self.df["target"].astype("int32")
        self.df['target_percent_pred'] = self.pred["target"][0]
        self.df['target_pred'] = self.pred.index[0]
        self.df["target_pred"] = self.df["target_pred"].astype("int32")
        self.df["target_percent_pred"] = pd.to_numeric(self.df["target_percent_pred"])

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
            
            print("---------------------->")
            print("\tpercentual " + str(i))
            pred = df["right"]
            print("---------------------->ALL DATA")
            print(pred.value_counts())
            print(pred.value_counts(normalize=True))
            print(df.loc[:, ["right", "target_pred", "target"]].groupby(['target', 'target_pred']).count())
            print("----------------------> TRADE DATA")

            try:
                self.print_money_return(i)
            except:
                print("...")
            print("\n")

    def print_money_return(self, percent: int):

        # parametros da plataforma
        multiplicador = 20  # multiplcador de $
        percentual = 0.12 # percentual minimo de lucro ou prejuizo da plataforma
        percentual_max = 0.40 # percentual minimo de lucro ou prejuizo da plataforma
        investimento = 10 # reais
        taxa = 0.006 # relacioando a -> investimento = 10 
        
        df_money = self.df[self.df["target_percent_pred"] >= (percent / 100)]
        df_money['money_return'] = df_money['money_return'] * multiplicador
        df_percent_range = df_money[((abs(df_money["money_return"]) > percentual) & (abs(df_money["money_return"]) < percentual_max )) ]
        trade_in = df_percent_range[df_percent_range["target"] != 0]#.count() # numero de trades efetuados
        
        print(trade_in)

        print(trade_in['right'].value_counts())
        print(trade_in['right'].value_counts(normalize=True))
        print(trade_in.loc[:, ["right", "target_pred", "target"]].groupby(['target', 'target_pred']).count())

        print("Total de trades: %s" % (trade_in['target'].count()))
        print("\nLucro/Prejuizo real ~= R$ : " + str(round(df_percent_range["money_return"].sum() * multiplicador * investimento - (trade_in['target'].count() * taxa), 2)))
        print("Lucro/Prejuizo sem taxas ~= R$ : " + str(round((df_percent_range["money_return"].sum() * multiplicador * investimento), 2)))
        print("Lucro/Prejuizo sem taxas e minimos/maximos ~= R$ : " + str(round((df_money["money_return"].sum() * multiplicador * investimento), 2)))
        print("----------------------")

    def print_index(self, index, size):
        print('\n------------------')
        print('index %s - size %s' %(index, size))
        print('------------------\n')
