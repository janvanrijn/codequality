import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_orig', type=str, default=os.path.expanduser('~/experiments/code_smells/blob.csv'))
    parser.add_argument('--data1', type=str, default=os.path.expanduser('~/experiments/generated/lu/pmd_blob.csv'))
    parser.add_argument('--data2', type=str, default=os.path.expanduser('~/experiments/generated/kku/blob.csv'))

    return parser.parse_args()


def run(args):
    df1 = pd.read_csv(args.data1).set_index(['Name', 'repository']).drop('CommitHash', axis=1)
    df2 = pd.read_csv(args.data2)
    df2['Name'] = df2['code_name']
    df2 = df2.set_index(['Name', 'repository']).drop(['code_name', 'path'], axis=1)
    if len(df1) != len(df2):
        raise ValueError('Diff in size %s vs %s' % (len(df1), len(df2)))
    df1 = df1.join(df2, how='left')

    disagree_1 = set()  # lu says pos, kky says false
    disagree_2 = set()
    for idx, row in df1.iterrows():
        if row['label'] != row['is_positive']:
            raise ValueError('Error in row: does not agree on label')
        if row['y_hat'] == True and row['pmd_code_smell'] == False:
            disagree_1.add(idx)
        elif row['y_hat'] == False and row['pmd_code_smell'] == True:
            disagree_2.add(idx)
    print(len(disagree_1), len(disagree_2))

    df_orig = pd.read_csv(args.data_orig).set_index(['Name', 'repository'])
    for key in disagree_2:
        subframe = df_orig.loc[key]
        subframe = subframe[['WMC', 'TCC', 'ATFD']]
        print(key[0], key[1])
        print(subframe)
        print(subframe.isnull().values.any(), subframe.isnull().values.all())


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
