class Config():
    def __init__(self, currency, name, config) -> None:
        self.name = name
        self.path = "data/treined/"
        self.market = { 
            "currency": currency, 
            "request": False 
        }

        self.model = {
            "name": "LSTM",
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
            "reduce": 7,
            "path": "/models/data_predict",
            "indicators": True
        }
