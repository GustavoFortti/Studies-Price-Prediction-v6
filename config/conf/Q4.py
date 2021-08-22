CONF = {  # val_loss: 0.5860
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
        "target": {"columns": ["High"], "categorical": 2, "description": ["0", "1"]},
    }
}