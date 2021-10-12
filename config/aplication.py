import os

class Config():
    def __init__(self, currency: str, question: int, model_type: int, time_ahead: int=0) -> None:
        option = { 1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5", 6: "Q6", 7: "Q7", 8: "Q8", 9: "Q9"}
        currency = currency
        name = currency + '_' + option[int(question)] + ("C" if (int(model_type) == 1) else "R") + '_TAH_' + str(time_ahead)
        print(name)

        TARGET = { 
            "columns": [["Close"], ["Open"], ["High"], ["Low"]], "description": [[0, 1], [0, 1, -1]]
        }

        target = TARGET['columns'][question - 1]
        description = TARGET["description"][0 if len(target) == 1 else 1]
        epochs = 10 if (model_type == 2) else 5
        timesteps = 34 if (model_type == 2) else 8

        self.config = {
            "name": name,
            "path": "./data/treined/" + currency + "/",
            "market": {
                "currency": currency, 
                "request": False 
            },
            "model": {
                "name": "LSTM",
                "model_type": model_type,
                "slice": 0.01,
                "LSTM": {
                    "epochs": epochs
                }
            },
            "data": {
                "time": "1D",
                "timesteps": timesteps,
                "time_ahead": time_ahead,
                "predict": {
                    "columns": ["Close", "High", "Low", "Open", "Volume"]
                },
                "target": {
                    "columns": target, "description": description
                },
                "reduce": 1,
                "path":  "./data/treined/" + currency + "/data_predict/",
                "indicators": True
            }
        }

        self.generate_structure()

    def generate_structure(self) -> None:
        path = self.config['path'] + self.config["name"]
        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")
            os.makedirs(path + "/config")
            if (not os.path.exists(self.config['data']['path'])):
                os.makedirs(self.config['data']['path'])

            f = open(path + '/config/aplication.py', 'w')
            f.write("CONF = " + str(self.config))
            f.close()