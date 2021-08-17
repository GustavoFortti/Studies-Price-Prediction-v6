CONF_1 = { # loss: 0.6299 - accuracy: 0.6437 - val_loss: 0.6396 - val_accuracy: 0.6279
    "name": "AAPL" + "_Q1",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": True
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 8
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

CONF_2 = { # val_loss: 0.9285
    "name": "AAPL" + "_Q2",
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
        "timesteps": 21,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}

CONF_3 = { # loss: 0.4486 - accuracy: 0.8090 - val_loss: 0.4443 - val_accuracy: 0.8033
    "name": "AAPL" + "_Q3",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": True
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 7
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Open"], "categorical": 2, "description": ["0", "1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}

CONF_4 = {  # val_loss: 0.5860
    "name": "AAPL" + "_Q4",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": True
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 6
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["High"], "categorical": 2, "description": ["0", "1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}

CONF_5 = {  # val_loss: 0.5878
    "name": "AAPL" + "_Q5",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": True
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 8
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Low"], "categorical": 2, "description": ["0", "1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}

CONF = CONF_3

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