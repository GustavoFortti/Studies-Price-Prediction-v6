CONF = { # loss: 0.4486 - accuracy: 0.8090 - val_loss: 0.4443 - val_accuracy: 0.8033
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
            "epochs": 5
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