CONF = {
    "name": "AAPL_Q1",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": True
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 10
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Close"], "categorical": 2, "description": ["0", "1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}

class config():
    def __init__(self, currency: str='AA') -> None:
        self.name = currency
        self.path = "data/treined/"
        self.market = { 
            "currency": currency, 
            "request": True 
        }

        self.model = {
            "name": "LTSM",
            "slice": 0.01,
            "LTSM": {
                "epochs": 10
            }
        }

        self.data = {
            "time": "1D",
            "timesteps": 8,
            "predict": {"columns": ["Date", "Close", "High", "Low", "Open"]},
            "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
            "reduce": 7,
            "path": "/models/data_predict"
        }

    def set_currency(self, currency):
        self.market['currency'] = currency
        self.name = currency