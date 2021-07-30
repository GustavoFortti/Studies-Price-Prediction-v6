import sys

import pandas as pd

from src.data_manager.data_generated import Data_generated

class Data_manager():
    def __init__(self, gen_data: bool, mode: str) -> None:
        
        data = Data_generated(mode)
        x = data.get_predictor()
        y = data.get_target()


        if (mode == 'td'): sys.exit()