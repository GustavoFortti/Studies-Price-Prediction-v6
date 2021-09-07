import os
import sys

import pandas as pd
import numpy as np

from src.reports.data_report import Data_report
from src.reports.pred_report import Pred_report
from src.reports.price_report import Price_report

pd.options.mode.chained_assignment = None 

class Report(Data_report, Pred_report, Price_report):
    def __init__(self, config: dict, scaler: object, mode: str) -> None:
        super().__init__(config, scaler, mode)

        self.scaler = scaler
        self.config = config
        self.mode = mode