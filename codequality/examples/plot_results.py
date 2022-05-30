#updated: May 30th, 2022
import argparse
import logging
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../data/performances.csv')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/plot'))
    parser.add_argument('--measures', nargs='+', type=str, default=['accuracy', 'precision', 'recall', 'f1_score'])

    return parser.parse_args()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_file)
    df = df.drop(['Unnamed: 0'], axis=1)
    logging.info('Dtypes: %s' % str(df.dtypes))
    plt.rc('font', family='serif')

    for smell_type in df['smell type'].unique():
        smell_name = smell_type.split('.')[0]
        results_smell = df.loc[df['smell type'] == smell_type]
        pd.set_option('display.max_columns', None)

        for measure in args.measures:
            fig1, ax1 = plt.subplots()
            ax1.set_xlabel('Severity Threshold')
            ax1.set_ylabel('Performance')

            # #Note:  results_smell['classifier'].unique() = ['DummyClassifier(random_state=0)' 'DecisionTreeClassifier(random_state=0)' 'RandomForestClassifier(random_state=0)' 'pmd classifier']
            for classifier in results_smell['classifier'].unique():
                results_classifier = results_smell.loc[results_smell['classifier'] == classifier].set_index(['severity_threshold'])
                x = results_classifier.index.to_numpy(dtype=float)
                y = results_classifier[measure].to_numpy(dtype=float)
                if np.any((y > 0)):
                    ax1.plot(x, y, '-+', label=classifier)

            ax1.legend(loc='lower right')
            # Use when easier-to-read labels is preferred.
            # ax1.legend(['Majority Class Classifier', 'Decision Tree', 'Random Forest', 'PMD Classifier'], loc='lower right')
            filename = os.path.join(args.output_dir, '%s_%s.pdf' % (smell_name, measure))
            plt.savefig(filename)
            plt.close()
            logging.info('Saved to: %s' % filename)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
