import numpy as np
import pandas as pd


class BlobModel(object):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        TCC_THRESHOLD = (100.0 / 3.0)  # Note discrepancy between PMD code and percentages
        if row['WMC'] >= 47 and row['TCC'] < TCC_THRESHOLD and row['ATFD'] > 5:
            return True
        else:
            return False

    def predict(self, df: pd.DataFrame) -> np.array:
        return df.apply(self.predict_row, axis=1)
