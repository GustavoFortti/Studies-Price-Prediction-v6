import numpy as np
from pandas import DataFrame

class Fibonacci():
    def __init__(self, data: DataFrame, params: dict={'n': 1, 'negative': False}):
        n = params['n']
        f_percentual = [261.8, 161.8, 100, 78.6, 61.8, 38.2, 23.6]
        if(params['negative']): f_percentual = [-i for i in f_percentual]
        self.values = [f_percentual[n] * i for i in data.values]

    def get_values(self):
        return self.values