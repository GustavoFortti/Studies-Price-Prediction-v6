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
        self.report = Report(self.config, self.scaler, self.mode)

        data = Data_manager(self.mode, index, self.report, self.config, self.scaler)
        
        if (mode == 'tr'): self.train(data)
        else: self.pred(data.get_x())

    def train(self, data: object) -> None:
        catalyst = self.model(self.config)
        create_model = catalyst.classification if (self.config['model']['type'] == 1) else catalyst.regression
        create_model(data, self.report)
        # catalyst.save()


    def pred(self, x: np.array) -> None:
        catalyst = self.model(self.config)
        pred = catalyst.predict(x)
        self.report.report_pred(pred)