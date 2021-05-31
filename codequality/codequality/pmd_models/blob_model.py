import codequality.pmd_models
import pandas as pd


class BlobModel(codequality.pmd_models.AbstractModel):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        if row['WMC'] >= 47 and row['TCC'] < 1.0 / 3.0 and row['ATFD'] > 5:
            return True
        else:
            return False
