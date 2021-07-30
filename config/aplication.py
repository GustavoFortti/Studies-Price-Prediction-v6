CONF = {
    "name": "T0001",
    "path": "/data/treined/",
    "market": "BRL=X",
    "model": {
        "name": "LTSM",
        "split": 120,
        "LTSM": {
            "opochs": 30
        }
    },
    "data": {
        "time": "1D",
        "timesteps": 8,
        "predict": {"columns": ["date_time", "Close", "High", "Low"]},
        "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
        "reduce": 5
    }
}