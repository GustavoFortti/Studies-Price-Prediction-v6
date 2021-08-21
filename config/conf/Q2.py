CONF = { # val_loss: 0.9285
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
            "epochs": 5
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