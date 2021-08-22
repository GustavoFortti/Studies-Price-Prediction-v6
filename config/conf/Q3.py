CONF = { # loss: 0.4486 - accuracy: 0.8090 - val_loss: 0.4443 - val_accuracy: 0.8033
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
        "target": {"columns": ["Open"], "categorical": 2, "description": ["0", "1"]},
    }
}