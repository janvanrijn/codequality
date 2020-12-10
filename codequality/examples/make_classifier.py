import argparse
import logging
import os
import pandas as pd
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))

    return parser.parse_args()


def evaluate(frame, type):
    print(frame.dtypes)
    frame = frame.drop([
        'severity',
        'CommitHashPrefix',
        'Name',
        'File'
    ], axis=1)
    frame['smell'] = frame['smell'].apply(lambda value: True if value == type else False)
    print(frame['smell'].describe())
    print(frame['smell'].value_counts())
    # logging.info('Class balance: %d/%d' % ())
    print(frame.head(5))


def run(args):
    files = os.listdir(args.input_dir)

    for idx, file in enumerate(files):
        file_name = os.path.splitext(file)[0]
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue

        file = os.path.join(args.input_dir, file)
        frame = pd.read_csv(file)
        evaluate(frame, file_name)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
