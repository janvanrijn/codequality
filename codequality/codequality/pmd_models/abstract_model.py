import numpy as np
import pandas as pd


class AbstractModel(object):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        raise NotImplementedError()

