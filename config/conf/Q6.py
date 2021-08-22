CONF = {  # val_loss: 0.5878
    "model": {
        "name": "LSTM",
        "slice": 0.01,
        "LSTM": {
            "epochs": 10
        }
    },
    "data": {
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["High", "Low"], "categorical": 3, "description": ["0", "1", "-1"]},
    }
}