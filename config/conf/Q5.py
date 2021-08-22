CONF = {  # val_loss: 0.5878
    "model": {
        "name": "LSTM",
        "slice": 0.01,
        "LSTM": {
            "epochs": 5
        }
    },
    "data": {
        "timesteps": 8,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Low"], "categorical": 2, "description": ["0", "1"]},
    }
}