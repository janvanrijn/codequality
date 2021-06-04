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
    parser.add_argument('--severity_threshold', type=float, default=0.75)

    return parser.parse_args()


def get_data_and_labels(frame: pd.DataFrame, severity_threshold: int):
    frame_prime = frame.groupby(['CommitHash', 'Name']).mean()
    logging.info("Size after grouping: (%d, %d)" % frame_prime.shape)

    y = frame['severity'].apply(lambda value: True if value > severity_threshold else False).to_numpy(dtype=bool)
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


def evaluate(data: np.array, labels: np.array, clf: sklearn.base.BaseEstimator,
             scorers: typing.List[str]):
    # TODO: evaluate according to: leave one project out
    # TODO: or on file level predictions
    classifier = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1.0),
        sklearn.feature_selection.VarianceThreshold(),
        clf
    )
    result = sklearn.model_selection.cross_validate(
        classifier, data, labels, scoring=scorers, cv=10)
    return result


def run(args):
    files = os.listdir(args.input_dir)
    random_seed = 0
    # precision / recall for binary targets
    scorers = {
        'accuracy': (sklearn.metrics.accuracy_score, {}),
        'precision': (sklearn.metrics.precision_score, {'zero_division': 0.0}), 
        'recall': (sklearn.metrics.recall_score, {})
    }
    scorers_sklearn = {
        k: sklearn.metrics.make_scorer(v[0], **v[1]) for k, v in scorers.items()
    }

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

        for clf in clfs:
            all_results = evaluate(data, labels, clf, scorers_sklearn)
            for scorer_name in scorers:
                result = all_results["test_%s" % scorer_name]
                logging.info("%s %s %s: %f +/- %f" % (
                    os.path.basename(file), str(clf), scorer_name,
                    float(np.mean(result)), float(np.std(result))))
        if filename == 'data class.csv':
            handmade = codequality.pmd_models.DataClassModel()
        elif filename == 'blob.csv':
            handmade = codequality.pmd_models.BlobModel()
        else:
            raise ValueError('not recognized file: %s' % file)
        y_hat = handmade.predict(frame)
        for scorer_name, (scorer_fn, kwargs) in scorers.items():
            logging.info("%s handmade %s: %f" % (os.path.basename(file), scorer_name, scorer_fn(labels, y_hat, **kwargs)))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
