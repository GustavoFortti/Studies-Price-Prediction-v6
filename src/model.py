from pathlib import Path
import pandas as pd

from config.aplication import CONF
from src.data_manager.data_manager import Data_manager

class Model():
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.generate_structure()

    def data(self, gen_data: bool=False):
        self.data = Data_manager(gen_data, self.mode)

    def train(self):
        pass

    def test(self, init, end):
        pass

    def pred(self):
        pass

    def generate_structure(self):
        Path(CONF['path'] + CONF['name'] + '/models').mkdir(parents=True, exist_ok=True)
        Path(CONF['path'] + CONF['name'] + '/trades').mkdir(parents=True, exist_ok=True)
        Path(CONF['path'] + CONF['name'] + '/notes.txt').touch(exist_ok=True)