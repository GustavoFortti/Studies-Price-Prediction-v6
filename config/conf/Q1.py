CONF = { # loss: 0.6299 - accuracy: 0.6437 - val_loss: 0.6396 - val_accuracy: 0.6279
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