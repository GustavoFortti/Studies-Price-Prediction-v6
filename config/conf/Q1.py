CONF = { # loss: 0.6299 - accuracy: 0.6437 - val_loss: 0.6396 - val_accuracy: 0.6279
    "model": {
        "name": "LSTM",
        "slice": 0.01,
        "LSTM": {
            "epochs": 360
        }
    },
    "data": {
        "timesteps": 20,
        "predict": {"columns": ["Close", "High", "Low", "Open", "Volume"]},
        "target": {"columns": ["Close"], "categorical": 2, "description": ["0", "1"]},
    }
}