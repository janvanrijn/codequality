import numpy as np
import pandas as pd


class DataClassModel(object):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        nopa = row['NOPA']
        noam = row['NOAM']
        wmd = row['WMC']
        if (nopa + noam > 3 and wmd < 31) or (nopa + noam > 5 and wmd < 47):
            return True
        else:
            return False

    def predict(self, df: pd.DataFrame) -> np.array:
        return df.apply(self.predict_row, axis=1)
