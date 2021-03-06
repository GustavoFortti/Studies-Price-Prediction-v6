import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

class Api_market():
    def __init__(self, mode: str, config: dict) -> None:
        currency = config['market']['currency']
        path = config['path']
        request = config['market']['request']

        file = path + currency + '.csv'
        if ((request) | (not os.path.isfile(file)) | (mode == 'pr')):
            data = yf.Ticker(currency)
            self.data = data.history(period="max")
            self.data = self.data.loc[:, config['data']['predict']['columns']]
            self.data = self.data.dropna()
            for i in self.data.columns: 
                self.data = self.data[self.data[i] != 0]
            self.data.to_csv(file)
        else:
            self.data = pd.read_csv(file, index_col='Date')
        

    def no_api(self):
        self.data = pd.read_csv('./data/EURUSD60.csv', names=["Date", "Open", "High", "Low", "Close", "Volume"], sep='\t')
        self.data = self.data.set_index('Date')
        return self.data

