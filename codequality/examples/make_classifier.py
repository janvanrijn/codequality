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
    parser.add_argument('--severity_thresholds', nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])

    return parser.parse_args()


def get_data_and_labels(frame: pd.DataFrame, severity_threshold: int):
    # Note that >= is important detail
    y = frame['severity'].apply(lambda value: True if value >= severity_threshold else False).to_numpy(dtype=bool)
    logging.info("Dtypes:\n" + str(frame.dtypes))
    logging.info("Values Count:\n" + str(frame['smell'].value_counts()))
    frame = frame.drop([
        'repository', 'package', 'filename', 'class_name', 'code_name', 'commit_hash', 'smell', 'severity'
    ], axis=1)
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Class distribution: %s" % str(dict(zip(unique, counts))))
    return frame.to_numpy(dtype=float), y


def evaluate_predictions(frame: pd.DataFrame, y_hat: np.array, filename: typing.Optional[str]) -> typing.Dict:
    scorers = {
        'accuracy': (sklearn.metrics.accuracy_score, {}),
        'precision': (sklearn.metrics.precision_score, {'zero_division': 0.0}),
        'recall': (sklearn.metrics.recall_score, {}),
        'confusion_matrix': (sklearn.metrics.confusion_matrix, {})
    }
    frame['y_hat'] = y_hat

    # Very important. Note that ['Name', 'CommitHash', 'repository'] are the keys from the create dataset script
    frame = frame[['code_name', 'commit_hash', 'repository', 'label', 'y_hat']].groupby(['code_name', 'commit_hash', 'repository']).agg([np.any])
    if filename is not None:
        frame.reset_index().to_csv(filename)
    logging.info('Frame size after aggregate: (%s,%s)' % frame.shape)
    logging.info("Values Count:\n" + str(frame['label'].value_counts()))
    scores = dict()
    for scorer_name, (scorer_fn, kwargs) in scorers.items():
        performance = scorer_fn(frame['label'], frame['y_hat'], **kwargs)
        logging.info("%s: %s" % (scorer_name, str(performance)))
        if isinstance(performance, np.ndarray):
            tn, fp, fn, tp = performance.ravel()
            scores['tn'] = tn
            scores['fp'] = fp
            scores['fn'] = fn
            scores['tp'] = tp
        else:
            scores[scorer_name] = performance
    return scores


def run(args):
    files = os.listdir(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    random_seed = 0

    clfs = [
        sklearn.dummy.DummyClassifier(random_state=random_seed),
        sklearn.tree.DecisionTreeClassifier(random_state=random_seed),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed, n_estimators=100)
    ]

    results = []
    for idx, file in enumerate(files):
        filename = os.path.basename(file)
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue
        logging.info("======= Smell Type: %s =======" % filename)

        file = os.path.join(args.input_dir, file)
        frame = pd.read_csv(file)
        for severity_threshold in args.severity_thresholds:
            logging.info("======= Severity Threshold: %s =======" % severity_threshold)
            data, labels = get_data_and_labels(frame, severity_threshold)
            frame['label'] = labels

            for clf in clfs:
                classifier = sklearn.pipeline.make_pipeline(
                    sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1.0),
                    sklearn.feature_selection.VarianceThreshold(),
                    clf
                )
                y_hat = sklearn.model_selection.cross_val_predict(classifier, data, labels, cv=10)
                logging.info('=== %s classifier ===' % clf)
                performance = evaluate_predictions(frame, y_hat, None)
                performance['severity_threshold'] = severity_threshold
                performance['classifier'] = str(clf)
                performance['smell type'] = filename
                results.append(performance)

            if filename == 'data class.csv':
                handmade = codequality.pmd_models.DataClassModel()
            elif filename == 'blob.csv':
                handmade = codequality.pmd_models.BlobModel()
            else:
                raise ValueError('not recognized file: %s' % file)
            logging.info('=== pmd classifier ===')
            y_hat = handmade.predict(frame)
            performance = evaluate_predictions(frame, y_hat, os.path.join(args.output_dir, 'pmd_%s' % filename))  # filename includes ext
            performance['severity_threshold'] = severity_threshold
            performance['classifier'] = 'pmd classifier'
            performance['smell type'] = filename
            results.append(performance)
    df = pd.DataFrame(results)
    output_file_csv = os.path.join(args.output_dir, "performances.csv")
    df.to_csv(output_file_csv)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
