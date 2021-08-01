import sys

import pandas as pd

from src.data_manager.data_generated import Data_generated

class Data_manager():
    def __init__(self, mode: str) -> None:
        
        data_gen = Data_generated(mode)
        x = data_gen.get_predictor()
        y = data_gen.get_target()

        print(x)
        print(y)

        if (mode == 'td'): sys.exit()