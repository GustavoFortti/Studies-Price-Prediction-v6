import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Price_report():
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        self.scaler = scaler
        self.config = config
        self.mode = mode