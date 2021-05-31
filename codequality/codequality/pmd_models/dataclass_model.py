import codequality.pmd_models
import pandas as pd


class DataClassModel(codequality.pmd_models.AbstractModel):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        nopa = row['NOPA']
        noam = row['NOAM']
        wmd = row['WMC']
        if (nopa + noam > 3 and wmd < 31) or (nopa + noam > 5 and wmd < 47):
            return True
        else:
            return False
