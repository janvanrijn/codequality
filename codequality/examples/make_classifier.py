import argparse
import logging
import numpy as np
import os
import pandas as pd
import typing

import codequality.pmd_models

import sklearn
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.pipeline
import sklearn.tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/generated/lu'))
    parser.add_argument('--severity_threshold', type=float, default=0.75)

    return parser.parse_args()


def get_data_and_labels(frame: pd.DataFrame, severity_threshold: int):
    # Note that >= is important detail
    y = frame['severity'].apply(lambda value: True if value >= severity_threshold else False).to_numpy(dtype=bool)
    logging.info("Dtypes:\n" + str(frame.dtypes))
    logging.info("Values Count:\n" + str(frame['smell'].value_counts()))
    frame = frame.drop([
        'severity',
        'CommitHash',
        'Name',
        'File',
        'repository',
        'smell'
    ], axis=1)
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Class distribution: %s" % str(dict(zip(unique, counts))))
    return frame.to_numpy(dtype=float), y


def evaluate_predictions(frame: pd.DataFrame, y_hat: np.array, filename: typing.Optional[str]):
    scorers = {
        'accuracy': (sklearn.metrics.accuracy_score, {}),
        'precision': (sklearn.metrics.precision_score, {'zero_division': 0.0}),
        'recall': (sklearn.metrics.recall_score, {}),
        'confusion_matrix': (sklearn.metrics.confusion_matrix, {})
    }
    frame['y_hat'] = y_hat

    # Very important. Note that ['Name', 'CommitHash', 'repository'] are the keys from the create dataset script
    frame = frame[['Name', 'CommitHash', 'repository', 'label', 'y_hat']].groupby(['Name', 'CommitHash', 'repository']).agg([np.any])
    if filename is not None:
        frame.reset_index().to_csv(filename)
    logging.info('Frame size after aggregate: (%s,%s)' % frame.shape)
    logging.info("Values Count:\n" + str(frame['label'].value_counts()))
    for scorer_name, (scorer_fn, kwargs) in scorers.items():
        performance = scorer_fn(frame['label'], frame['y_hat'], **kwargs)
        logging.info("%s: %s" % (scorer_name, str(performance)))


def run(args):
    files = os.listdir(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    random_seed = 0

    clfs = [
        sklearn.dummy.DummyClassifier(random_state=random_seed),
        sklearn.tree.DecisionTreeClassifier(random_state=random_seed),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed, n_estimators=100)
    ]

    for idx, file in enumerate(files):
        filename = os.path.basename(file)
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue
        logging.info("======= %s =======" % filename)

        file = os.path.join(args.input_dir, file)
        frame = pd.read_csv(file)
        data, labels = get_data_and_labels(frame, args.severity_threshold)
        frame['label'] = labels

        for clf in clfs:
            classifier = sklearn.pipeline.make_pipeline(
                sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1.0),
                sklearn.feature_selection.VarianceThreshold(),
                clf
            )
            y_hat = sklearn.model_selection.cross_val_predict(classifier, data, labels, cv=10)
            logging.info('=== %s classifier ===' % clf)
            evaluate_predictions(frame, y_hat, None)

        if filename == 'data class.csv':
            handmade = codequality.pmd_models.DataClassModel()
        elif filename == 'blob.csv':
            handmade = codequality.pmd_models.BlobModel()
        else:
            raise ValueError('not recognized file: %s' % file)
        logging.info('=== pmd classifier ===')
        y_hat = handmade.predict(frame)
        evaluate_predictions(frame, y_hat, os.path.join(args.output_dir, 'pmd_%s' % filename))  # filename includes ext


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
