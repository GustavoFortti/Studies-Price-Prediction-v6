from copy import deepcopy
from os import name
import numpy as np
import pandas as pd
import ta

from src.data_manager.indicators_analysis.generate_labels import Genlabels
from src.data_manager.indicators_analysis.poly_interpolation import PolyInter
from src.data_manager.indicators_analysis.sar_parabolic import Parabolic_sar
from src.data_manager.indicators_analysis.date_time import Date_time

class Inticators_manager():
    def __init__(self, is_predict: bool, config: dict) -> None:
        self.config = config
        self.is_predict = is_predict

    def generate(self, df) -> pd.DataFrame:
        df = deepcopy(df)
        if (self.is_predict):
            return self.prediction(df)
            # return df
        else:
            return self.target(df)
            
    def target(self, df) -> pd.DataFrame:
        if (self.config['model']['type'] == 1):
            ax_df = df.loc[:, self.config['data']['target']['columns']]
            ax_df = self.convert_col_to_bool(ax_df, ax_df.columns)
            if (len(ax_df.columns) >= 2): ax_df = self.cross_bool_cols(ax_df, [ax_df.columns])
            return pd.DataFrame(np.array(ax_df), columns=['target'], index=df.index)
        
        df = df.loc[:, self.config['data']['target']['columns']].append(pd.DataFrame(np.array([0]), columns=self.config['data']['target']['columns'])).iloc[1:, :]
        df.columns = ['target']
        return df

    def prediction(self, df) -> pd.DataFrame:
        indicators = [
            {"name": "labels", "columns": ['Close', 'Open', 'High', 'Low'], "method": Genlabels, "params": {"window": 25, "polyorder": 3}},
            {"name": "PolyInter", "columns": ['Close', 'Open', 'High', 'Low'], "method": PolyInter, "params": {"degree":4, "pd":20, "plot":False, "progress_bar":True}},
        ]
    
        df = ta.add_all_ta_features(df=df, close="Close", high='High', low='Low', open="Open", volume="Volume", fillna=True)
        df = self.convert_col_to_bool(df, ['Close', 'High', 'Low', 'Open'])
        df = self.indicators_analysis(df, indicators)
        df = self.col_parabolic_sar(df, ['High', 'Low'], False)
        df = self.col_date(df)
        
        columns_cross = [['High_bool', 'Low_bool'], ['Close_bool', 'Open_bool'], ['High_bool', 'Low_bool', 'Close_bool', 'Open_bool']]
        df = self.cross_bool_cols(df, columns_cross) 
        
        return df

    def convert_col_to_bool(self, df, cols) -> pd.DataFrame:
        df = deepcopy(df)
        to_left = 0 if not self.is_predict else -1

        for i in cols:
            col = np.array(df[i])
            size = len(df[i])
            res = [] if not self.is_predict else [0]

            for j in range(size + to_left):
                if (j < size - 1):
                    if (col[j] > col[j + 1]): res.append(0)
                    else: res.append(1)
                else:
                    res.append(None)
            if (not self.is_predict): df[i] = res
            else: df[i + '_bool'] = res

        return df        

    def cross_bool_cols(self, df, cols) -> pd.DataFrame:
        df = deepcopy(df)
        for i in cols:
            ax_df = [ 1 if j == len(i) else -1 if j == 0 else 0 for j in df.loc[:, i].sum(axis=1)]
            if (self.is_predict): df["__".join(i)] = ax_df
            else: df = ax_df

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
        return df

