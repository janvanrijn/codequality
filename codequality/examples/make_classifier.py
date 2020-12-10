import argparse
import logging
import os
import pandas as pd
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))

    return parser.parse_args()


def evaluate(frame):
    print(frame)


def run(args):
    files = os.listdir(args.matrices_dir)

    for idx, file in enumerate(files):
        file = os.path.join(args.matrices_dir, file)
        frame = pd.read_csv(file)
        evaluate(frame)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
