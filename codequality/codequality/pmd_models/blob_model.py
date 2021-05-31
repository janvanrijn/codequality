import logging
import numpy as np
import pandas as pd
import sklearn.metrics
import typing


class BlobModel(object):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        if row['CountDeclMethodDefault'] > 1.0:
            return False
        else:
            return True

    @staticmethod
    def predict(df: pd.DataFrame) -> typing.List[bool]:
        return df.apply(BlobModel.predict_row, axis=1)

    @staticmethod
    def evaluate(df: pd.DataFrame, labels: np.array) -> typing.Tuple[float, np.array]:
        # returns performance and the indices of the mistakes
        y_hat = BlobModel.predict(df)
        score = sklearn.metrics.accuracy_score(labels, y_hat)
        df = df[y_hat != labels]
        logging.info('BlobModel: Accuracy %f, mistakes %d' % (score, len(df)))
        return score, y_hat != labels
