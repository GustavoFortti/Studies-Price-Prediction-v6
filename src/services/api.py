import os
from datetime import datetime

from config.aplication import CONF

import pandas as pd
import yfinance as yf

class Api_market():
    def __init__(self) -> None:
        currency = CONF['market']['currency']
        path = CONF['path'] + CONF['name']
        request = CONF['market']['request']

        file = "%s/%s.csv" % (path, currency)
        if ((request) | (not os.path.isfile(file))):
            data = yf.Ticker(currency)
            self.data = data.history(period="max")
            self.data.to_csv(file)
        else:
            self.data = pd.read_csv(file, index_col='Date')

        self.data = self.data.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
            
class Api_trade():
    def __init__(self, market) -> None:
        self.data = yf.Ticker(market)
        
    def test(self):
        pass

    def play(self):
        pass
        # while (True):
        #     print(datetime.now())
        #     if ((datetime.now().strftime("%M") == "59") & (datetime.now().strftime("%S") in ["30", "31", "32"])):
        #         pass