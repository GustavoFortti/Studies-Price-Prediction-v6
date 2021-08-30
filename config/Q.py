CONF_1V1 = {
    "model": {
        "name": "LSTM",
        "slice": 0.01,
        "LSTM": {
            "epochs": 5
        }
    },
    "data": {
        "timesteps": 8,
        "indicators": "",
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Close"], "categorical": 2, "description": ["0", "1"]},
    }
}
