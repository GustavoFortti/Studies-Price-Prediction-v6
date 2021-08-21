CONF = {  # val_loss: 0.5878
    "name": "AAPL" + "_Q6",
    "path": "data/treined/",
    "market": {
        "currency": "AAPL",
        "request": False
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 10
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
        "reduce": 7,
        "path": "/models/data_predict"
    }
}