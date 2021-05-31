import logging
import numpy as np
import pandas as pd
import sklearn.metrics
import typing


class AbstractModel(object):

    @staticmethod
    def predict_row(row: pd.Series) -> bool:
        raise NotImplementedError()

    def predict(self, df: pd.DataFrame) -> typing.List[bool]:
        return df.apply(self.predict_row, axis=1)

    def evaluate(self, df: pd.DataFrame, labels: np.array) -> typing.Tuple[float, np.array]:
        # returns performance and the indices of the mistakes
        y_hat = self.predict(df)
        score = sklearn.metrics.accuracy_score(labels, y_hat)
        df = df[y_hat != labels]
        logging.info('AbstractModel: Accuracy %f, mistakes %d' % (score, len(df)))
        return score, y_hat != labels
