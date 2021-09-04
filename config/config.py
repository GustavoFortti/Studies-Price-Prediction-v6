import os

TARGET = { 
   "columns": [["Close"], ["Open"], ["High"], ["Low"], ["High", "Low"]], "description": [["0", "1"], ["0", "1", "-1"]]
}

class Config_2():
    def __init__(self, currency, name, question, _type) -> None:
        target = TARGET['columns'][question - 1]
        description = TARGET["description"][0 if len(target) == 1 else 1]
        epochs = 80 if (target == 2) else 5
        timesteps = 20 if (target == 2) else 8

        self.config = {
            "name": name,
            "path": "./data/treined/" + currency + '/',
            "market": {
                "currency": currency, 
                "request": False 
            },
            "model": {
                "name": "LSTM",
                "type": _type,
                "slice": 1,
                "LSTM": {
                    "epochs": epochs
                }
            },
            "data": {
                "time": "1D",
                "timesteps": timesteps,
                "predict": {
                    "columns": ["Close", "High", "Low", "Open", "Volume"]
                },
                "target": {
                    "columns": target, "description": description
                },
                "reduce": 1,
                "path": "/models/data_predict/",
                "indicators": True
            }
        }
        print(self.config)
        self.generate_structure()

    def generate_structure(self) -> None:
        path = self.config['path'] + self.config["name"]
        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")
            os.makedirs(path + "/models/data_predict/")
            os.makedirs(path + "/config")

            f = open(path + '/config/aplication.py', 'w')
            f.write("CONF = " + str(self.config))
            f.close()