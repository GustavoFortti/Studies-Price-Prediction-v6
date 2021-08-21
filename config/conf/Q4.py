CONF = {  # val_loss: 0.5860
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
            "epochs": 5
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