import os

import pandas as pd

from config.aplication import CONF
from src.data_manager.data_manager import Data_manager

class Model():
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.generate_structure()
        self.data = Data_manager(self.mode)

    def train(self):
        pass

    def test(self):
        pass

    def pred(self):
        pass

    def generate_structure(self):
        path = CONF['path'] + CONF['name']

        if (not os.path.exists(path)):
            os.makedirs(path)
            os.makedirs(path + "/models")