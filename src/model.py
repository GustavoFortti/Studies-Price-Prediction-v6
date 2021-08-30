import os
import sys

import numpy as np
import pandas as pd
from tensorflow.python.framework import config

from src.reports.report import Report
from src.models.LSTM.lstm import LTSM_model
from src.data_manager.data_manager import Data_manager
from sklearn.preprocessing import StandardScaler

class Model():
    def __init__(self, config: dict, mode: str, index: int=1) -> None:
        self.scaler = StandardScaler() 
        self.config = config
        self.model = LTSM_model
        self.report = Report(config)
        self.mode = mode
        self.generate_structure()

        data = Data_manager(self.mode, index, self.report, self.config, self.scaler)

        if (mode == 'tr'):
            x_train, x_test, y_train, y_test = data.get_train_test()
            self.train(x_train, x_test, y_train, y_test)
        elif (mode == 'te'):
            self.test(data.get_x())
        elif (mode == 'pr'):
            self.pred(data.get_x())

    def train(self, x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array) -> None:
        catalyst = self.model(self.config)

        create_model = catalyst.classification if (self.config.model['type'] == 1) else catalyst.regression
        create_model(x_train, x_test, y_train, y_test)
        catalyst.save()

    def test(self, x: np.array) -> None:
        catalyst = self.model(self.config)
        pred = catalyst.predict(x)
        self.print_resp(pred)

    def pred(self, x: np.array) -> None:
        catalyst = self.model(self.config)
        pred = catalyst.predict(x)
        self.print_resp(pred)

    def print_resp(self, pred):
        if (self.config.model['type'] == 1):
            ax_df = pd.DataFrame(pred, columns=self.config.data["target"]["description"])
            ax_df= ax_df.T
            ax_df.columns = ["target"]
            out = ax_df.sort_values(by="target", ascending=False)
            print(out)
        else: 
            out = self.scaler.inverse_transform(pred)
            print(out)

        f = open("./notebooks/out.txt", 'a')
        f.write(self.config.name + ' - ' + str(out) + ' - ' + str(self.config.data['target']['columns']) + '\n')

    def generate_structure(self) -> None:
        path = self.config.path + self.config.name

        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")
            os.makedirs(path + "/config")

            f = open(path + '/config/aplication.py', 'w')
            f.write("CONF = " + str(self.config))
            f.close()