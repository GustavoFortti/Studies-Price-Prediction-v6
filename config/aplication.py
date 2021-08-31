class Config():
    def __init__(self, currency, name, config, _type) -> None:
        self.name = name
        self.path = "data/treined/" + currency + '/'
        self.market = { 
            "currency": currency, 
            "request": False 
        }

        self.model = {
            "name": "LSTM",
            "type": _type, # 1 = classification, 2 = regression
            "slice": config["model"]["slice"],
            "LSTM": {
                "epochs": config["model"]["LSTM"]["epochs"]
            }
        }

        self.data = {
            "time": "1D",
            "timesteps": config["data"]["timesteps"],
            "predict": config["data"]["predict"],
            "target": config["data"]["target"],
            "reduce": 1,
            "path": "/models/data_predict",
            "indicators": True
        }
