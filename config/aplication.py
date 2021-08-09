CONF = {
    "name": "T0001",
    "path": "data/treined/",
    "market": {
        "currency": "EURUSD=X",
        "request": False
    },
    "model": {
        "name": "LTSM",
        "slice": 0.01,
        "LTSM": {
            "epochs": 30
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["Date", "Close", "High", "Low", "Open"]},
        "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
        "reduce": 5,
        "path": "/models/data"
    }
}