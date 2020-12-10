import argparse
import logging
import os
import pandas as pd

import sklearn
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))

    return parser.parse_args()


def get_data_and_labels(frame: pd.DataFrame, smell_type: str, random_seed: int):
    frame['smell'] = frame['smell'].apply(lambda value: True if value == smell_type else False)

    df_majority = frame[~frame['smell']]
    df_minority = frame[frame['smell']]
    df_majority_downsampled = sklearn.utils.resample(
        df_majority, replace=True, n_samples=len(df_minority),
        random_state=random_seed)
    frame = pd.concat([df_majority_downsampled, df_minority])

    y = frame['smell'].to_numpy(dtype=bool)

    logging.info("Dtypes:\n" + str(frame.dtypes))
    logging.info("Values Count:\n" + str(frame['smell'].value_counts()))
    frame = frame.drop([
        'severity',
        'CommitHashPrefix',
        'Name',
        'File',
        'smell'
    ], axis=1)
    return frame.to_numpy(dtype=float), y


def evaluate(frame: pd.DataFrame, smell_type: str, random_seed: int):
    data, labels = get_data_and_labels(frame, smell_type)
    classifier = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1.0),
        sklearn.feature_selection.VarianceThreshold(),
        sklearn.ensemble.RandomForestClassifier(random_state=random_seed)
    )
    result = sklearn.model_selection.cross_val_score(classifier, data, labels)
    return result


def run(args):
    files = os.listdir(args.input_dir)
    random_seed = 0

    for idx, file in enumerate(files):
        file_name = os.path.splitext(file)[0]
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue

        file = os.path.join(args.input_dir, file)
        frame = pd.read_csv(file)
        result = evaluate(frame, file_name, random_seed)
        print(result)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
