import yfinance as yf
from datetime import datetime

class Api_market():
    def __init__(self, market) -> None:
        self.data = yf.Ticker(market)
    
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