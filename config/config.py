CONF = { 
    "model": {
        "name": "LSTM",
        "LSTM": {
            "epochs": 35
        }
    },
    "data": {
        "timesteps": 20,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": [["Close"], ["High"], ["Low"], ["Open"], ["High", "Low"]], "description": [["0", "1"], ["0", "1", "-1"]]},
    }
}

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
            "type": _type,
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
