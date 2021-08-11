import os
import sys

import numpy as np
import pandas as pd

from config.aplication import CONF
from src.reports.report import Report
from src.models.LTSM.ltsm import LTSM_model
from src.data_manager.data_manager import Data_manager

class Model():
    def __init__(self, mode: str, index: int=1) -> None:
        self.model = LTSM_model
        self.report = Report()
        self.mode = mode
        self.generate_structure()
        data = Data_manager(self.mode, index, self.report)

        if (mode == 'tr'):
            x_train, x_test, y_train, y_test = data.get_train_test()
            self.train(x_train, x_test, y_train, y_test)
        elif (mode == 'te'):
            self.test(data.get_x())
        elif (mode == 'pr'):
            self.pred(data.get_x())

    def train(self, x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array) -> None:
        catalyst = self.model()
        catalyst.create(x_train, x_test, y_train, y_test)
        catalyst.save()

    def test(self, x: np.array) -> None:
        catalyst = self.model()
        pred = catalyst.predict(x)

        self.report.set_pred(pred)
        self.report.test_model_report()
        self.report.print_analisys()

    def pred(self, x: np.array) -> None:
        catalyst = self.model()
        pred = catalyst.predict(x)

    def generate_structure(self) -> None:
        path = CONF['path'] + CONF['name']

        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")
            os.makedirs(path + "/models/data_predict")
            os.makedirs(path + "/config")

            f = open(path + '/config/aplication.py', 'w')
            f.write("CONF = " + str(CONF))
            f.close()