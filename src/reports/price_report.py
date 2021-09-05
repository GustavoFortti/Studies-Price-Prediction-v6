import os
import sys

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

class Price_report():
    def __init__(self, config: dict, scalar: object, mode: str) -> None:
        self.scalar = scalar
        self.config = config
        self.mode = mode