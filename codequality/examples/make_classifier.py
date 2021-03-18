import argparse
import logging
import numpy as np
import os
import pandas as pd
import typing

import sklearn
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))

    return parser.parse_args()


def get_data_and_labels(frame: pd.DataFrame, severity_labels: typing.List[str]):
    y = frame['severity'].apply(lambda value: True if value in severity_labels else False).to_numpy(dtype=bool)
    logging.info("Dtypes:\n" + str(frame.dtypes))
    logging.info("Values Count:\n" + str(frame['smell'].value_counts()))
    frame = frame.drop([
        'severity',
        'CommitHash',
        'Name',
        'smell'
    ], axis=1)
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Class distribution: %s" % str(dict(zip(unique, counts))))
    return frame.to_numpy(dtype=float), y


def evaluate(frame: pd.DataFrame, scorers: typing.List[str], severity_labels: typing.List[str], random_seed: int):
    data, labels = get_data_and_labels(frame, severity_labels)
    classifier = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1.0),
        sklearn.feature_selection.VarianceThreshold(),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed)
    )
    result = sklearn.model_selection.cross_validate(
        classifier, data, labels, scoring=scorers, cv=10)
    return result


def run(args):
    files = os.listdir(args.input_dir)
    random_seed = 0
    # precision / recall for binary targets
    scorers = ['accuracy', 'precision', 'recall']

    for idx, file in enumerate(files):
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue

        file = os.path.join(args.input_dir, file)
        frame = pd.read_csv(file)
        all_results = evaluate(frame, scorers, ['minor', 'major', 'critical'], random_seed)
        for scorer in scorers:
            result = all_results["test_%s" % scorer]
            logging.info("%s: %f +/- %f" % (
                scorer, float(np.mean(result)), float(np.std(result))))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
