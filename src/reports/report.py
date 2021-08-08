import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Report():
    def __init__(self) -> None:
        pass

    def set_df_origin(self, x, y):
        x['target_origin'] = y['target']
        self.df = x

    def set_df_end(self, x, y, index):
        ax_array = x[-(1 + index):-(index)][0][-1:][0]
        ax_array = np.append(ax_array, y[-(1 + index):-(index)][0][0])

        for i, j in zip(self.df, ax_array):
            if (self.df[i].values == j): 
                print("Error target: unaligned data")
                sys.exit()

    def validation_data(): # função de validação dos dados
        pass

    def compare_train(): # função de comparação do treinamento com outros trinos
        pass

    def calc_test(): # função sobre todos os calculos de testes
        pass

    def pred():
        pass