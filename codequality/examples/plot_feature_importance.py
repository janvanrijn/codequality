import argparse
import logging
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data/importances_0.750000_blob.csv')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/plot'))
    parser.add_argument('--num_feats', type=int, default=10)

    return parser.parse_args()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    smell_name = os.path.basename(args.input_file).split('_')[-1]
    df = pd.read_csv(args.input_file).sort_values(by='importance', axis=0, ascending=False).head(n=args.num_feats).set_index('column')
    print(df)
    plt.rc('font', family='serif')
    fig1, ax1 = plt.subplots()
    df['importance'].plot.bar(yerr=df['std'].to_numpy(), ax=ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean decrease in impurity')

    ax1.set_xticklabels(df.index, rotation=45, ha='right')
    filename = os.path.join(args.output_dir, 'importances_%s.pdf' % smell_name)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info('Saved to: %s' % filename)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
