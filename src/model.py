import os

import numpy as np
import pandas as pd

from config.aplication import CONF
from src.models.LTSM.ltsm import LTSM_model
from src.data_manager.data_manager import Data_manager

class Model():
    def __init__(self, mode: str, index: int=0) -> None:
        self.mode = mode
        self.generate_structure()
        data = Data_manager(self.mode, index)

        if (mode == 'tr'):
            x_train, x_test, y_train, y_test = data.get_train_test()
            self.train(x_train, x_test, y_train, y_test)
        elif (mode == 'te'):
            self.test(data.get_x(), data.get_y())
        elif (mode == 'pr'):
            self.pred(data.get_x(), data.get_x_prove())

    def train(self, x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array) -> None:
        catalyst = LTSM_model()
        catalyst.create(x_train, x_test, y_train, y_test)
        catalyst.save()

    def test(self, x: np.array, y: np.array) -> None:
        catalyst = LTSM_model()
        pred = catalyst.predict(x)

    def pred(self, x: np.array) -> None:
        catalyst = LTSM_model()
        pred = catalyst.predict(x)

    def generate_structure(self) -> None:
        path = CONF['path'] + CONF['name']

        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")
            os.makedirs(path + "/config")

            f = open(path + '/config/aplication.py', 'w')
            f.write("CONF = " + str(CONF))
            f.close()