import sys

import numpy as np

from src.reports.report import Report
from src.models.LSTM.lstm import LTSM_model
from src.data_manager.data_manager import Data_manager
from sklearn.preprocessing import StandardScaler

class Model():
    def __init__(self, config: dict, mode: str, index: int=1) -> None:
        self.config = config
        self.mode = mode

        self.scaler = {"predictor": StandardScaler(), "target": StandardScaler()}
        self.model = LTSM_model
        self.report = Report(config, self.scaler, mode)

        data = Data_manager(self.mode, index, self.report, self.config, self.scaler)

        if (mode == 'tr'):
            x_train, x_test, y_train, y_test = data.get_train_test()
            self.train(x_train, x_test, y_train, y_test)
        else: self.pred(data.get_x())

    def train(self, x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array) -> None:
        catalyst = self.model(self.config)

        create_model = catalyst.classification if (self.config['model']['type'] == 1) else catalyst.regression
        create_model(x_train, x_test, y_train, y_test)
        catalyst.save()

    def pred(self, x: np.array) -> None:
        catalyst = self.model(self.config)
        pred = catalyst.predict(x)
        self.report.report_pred(pred)