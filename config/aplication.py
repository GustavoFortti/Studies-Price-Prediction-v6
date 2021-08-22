from config.conf.Q1 import CONF as Q1 # close - loss: 0.5979 - accuracy: 0.6698 - val_loss: 0.6365 - val_accuracy: 0.6360
from config.conf.Q2 import CONF as Q2 # high low - loss: 0.9153 - accuracy: 0.5889 - val_loss: 0.9202 - val_accuracy: 0.5894
from config.conf.Q3 import CONF as Q3 # open - loss: 0.3844 - accuracy: 0.8276 - val_loss: 0.4483 - val_accuracy: 0.8020
from config.conf.Q4 import CONF as Q4 # high - loss: 0.5530 - accuracy: 0.7248 - val_loss: 0.5764 - val_accuracy: 0.6991
from config.conf.Q5 import CONF as Q5 # low - loss: 0.5291 - accuracy: 0.7335 - val_loss: 0.5788 - val_accuracy: 0.6953
from config.conf.Q6 import CONF as Q6 # test

CONF = Q2

class Config():
    def __init__(self, currency, name) -> None:
        self.name = name
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
            "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
            "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
            "reduce": 7,
            "path": "/models/data_predict",
            "indicators": True
        }
