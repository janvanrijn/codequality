import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/Downloads/MLCQ_software_matrices/'))

    return parser.parse_args()


def run(args):
    for file in os.listdir(args.matrices_dir):
        logging.info("opening file: %s" % file)
        file_prefix = os.path.splitext(file)[0]
        file_splitted = file_prefix.split('-')

        frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        frame = frame[frame['Kind'] == 'Public Class'].drop('Kind', axis=1)
        frame['Project'] = '-'.join(file_splitted)
        frame['CommitHashPrefix'] = file_splitted[-1]
        frame = frame.set_index(['Name', 'Project', 'CommitHashPrefix'])


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())

