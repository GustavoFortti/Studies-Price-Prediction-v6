from copy import deepcopy
import numpy as np
import pandas as pd
import ta

from config.aplication import CONF

from src.data_manager.indicators_analysis.generate_labels import Genlabels
from src.data_manager.indicators_analysis.macd import Macd
from src.data_manager.indicators_analysis.rsi import StochRsi
from src.data_manager.indicators_analysis.dpo import Dpo
from src.data_manager.indicators_analysis.coppock import Coppock
from src.data_manager.indicators_analysis.poly_interpolation import PolyInter
from src.data_manager.indicators_analysis.sar_parabolic import Parabolic_sar
from src.data_manager.indicators_analysis.date_time import Date_time

class Inticators_manager():
    def __init__(self, _type) -> None:
        self.type = _type

    def generate(self, df) -> pd.DataFrame:
        df = deepcopy(df)
        if (self.type == 'predictor'):
            return self.prediction(df)
        else:
            df = self.target(df.loc[:, CONF['data']['target']['columns']])
            return self.cross_bool_cols(df, [df.columns])

    def target(self, df) -> pd.DataFrame:
        return self.convert_col_to_bool(df, df.columns)

    def prediction(self, df) -> pd.DataFrame:

        indicators = [
            {"name": "labels", "columns": ['High', 'Low'], "method": Genlabels, "params": {"window": 25, "polyorder": 3}},
            {"name": "Macd", "columns": ['High', 'Low'], "method": Macd, "params": {'short_pd':12, 'long_pd':26, 'sig_pd':9}},
            {"name": "StochRsi", "columns": ['High', 'Low'], "method": StochRsi, "params": {"period":14}},
            {"name": "Dpo", "columns": ['High', 'Low'], "method": Dpo, "params": {"period":4}},
            {"name": "Coppock", "columns": ['High', 'Low'], "method": Coppock, "params": {"wma_pd":10, "roc_long":6, "roc_short":3}},
            {"name": "PolyInter", "columns": ['High', 'Low'], "method": PolyInter, "params": {"degree":4, "pd":20, "plot":False, "progress_bar":True}},
        ]

        df = ta.add_all_ta_features(df=df, close="Close", high='High', low='Low', open="Open", volume="Volume", fillna=True)
        df = self.convert_col_to_bool(df, ['Close', 'High', 'Low'])
        df = self.indicators_analysis(df, indicators)
        df = self.col_parabolic_sar(df, ['High', 'Low'], False)
        df = self.col_date(df)
        
        columns_cross = [['High_bool', 'Low_bool']]
        df = self.cross_bool_cols(df, columns_cross) 

        return df

    def convert_col_to_bool(self, df, cols) -> pd.DataFrame:
        df = deepcopy(df)
        to_left = 0 if self.type == 'target' else -1

        for i in cols:
            col = np.array(df[i])
            size = len(df[i])
            res = [] if self.type == 'target' else [0]

            for j in range(size + to_left):
                if (j < size - 1):
                    if (col[j] > col[j + 1]): res.append(0)
                    else: res.append(1)
                else:
                    res.append(None)
            if (self.type == 'target'): df[i] = res
            else: df[i + '_bool'] = res

        return df        

    def cross_bool_cols(self, df, cols) -> pd.DataFrame:
        df = deepcopy(df)
        for i in cols:
            ax_df = [ 1 if j == len(i) else -1 if j == 0 else 0 for j in df.loc[:, i].sum(axis=1)]
            if (self.type != 'target'): df["__".join(i)] = ax_df
            else: df = pd.DataFrame(ax_df, columns=['target'], index=df.index)

        return df

    def indicators_analysis(self, df, indicators) -> pd.DataFrame:
        df = deepcopy(df)
        for i in indicators:
            for j in i['columns']:
                df[(j + '_' + i['name'])] = i['method'](df[j], i['params']).get_values()
        return df

    def col_parabolic_sar(self, df, cols, bool_col, params={"af":0.02, "amax":0.2}, name='parabolic_sar') -> pd.DataFrame:
        df = deepcopy(df)
        df[name] = Parabolic_sar(df.loc[:, cols], params, cols[0], cols[1]).values
        return df

    def col_date(self, df) -> pd.DataFrame:
        df = deepcopy(df)
        date = Date_time(df)
        df['weekday'] = date.Date()
        df['hour'] = date.Time()
        return df

