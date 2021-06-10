import argparse
import glob
import json
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_understand_dir', type=str, default=os.path.expanduser('~/data/codequality/understand_matrices'))
    parser.add_argument('--matrices_pmd_dir', type=str, default=os.path.expanduser('~/data/codequality/PMD_matrices_processed'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))
    parser.add_argument('--code_smells_path', type=str, default='../data/code_smells.csv')
    parser.add_argument('--included_projects', type=str, default='../data/included_projects.csv')
    parser.add_argument('--max_projects', type=int, default=None)
    parser.add_argument('--smell_type', type=str, default='data class')

    return parser.parse_args()


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    all_code_smells_frame = pd.read_csv(args.code_smells_path)
    all_code_smells_frame = all_code_smells_frame[all_code_smells_frame['smell'] == args.smell_type]

    def rewrite_strategy(val: str):
        if val == 'none':
            return 0
        elif val == 'minor':
            return 1
        elif val == 'major':
            return 2
        elif val == 'critical':
            return 3
        else:
            raise ValueError('severity value: %s' % val)

    all_code_smells_frame['severity'] = all_code_smells_frame['severity'].apply(rewrite_strategy)
    all_code_smells_frame['Name'] = all_code_smells_frame['code_name']
    all_code_smells_frame['filename'] = all_code_smells_frame['path'].apply(lambda l: os.path.splitext(os.path.basename(l))[0])
    all_code_smells_frame['package'] = all_code_smells_frame.apply(lambda row: row['code_name'][0:row['code_name'].rfind(row['filename']) - 1], axis=1)
    all_code_smells_frame = all_code_smells_frame[['repository', 'commit_hash', 'Name', 'package', 'filename', 'smell', 'severity']]
    all_code_smells_frame = all_code_smells_frame.groupby(['repository', 'commit_hash', 'Name', 'package', 'filename', 'smell']).mean()
    all_code_smells_frame = all_code_smells_frame.reset_index()
    logging.info("Number of records: %d" % len(all_code_smells_frame))
    original_frame_len = len(all_code_smells_frame)
    pmd_duplicate_rows = 0

    included_projects = pd.read_csv(args.included_projects)
    included_projects = included_projects['repository']

    missing_understand = set()
    missing_pmd = set()
    used_projects = set()

    list_projects_frames = []
    for idx, project_repo in enumerate(included_projects):
        if args.max_projects is not None and idx > args.max_projects:
            break
        logging.info("processing project: %s (%d/%d)" % (project_repo, idx+1, len(included_projects)))
        project_code_smells = all_code_smells_frame[all_code_smells_frame['repository'] == project_repo]
        commit_hash = project_code_smells['commit_hash'].iloc[0]
        project_code_smells = project_code_smells.set_index(['Name', 'repository'])
        logging.info("code smells: %d" % len(project_code_smells))

        missing = False
        understand_filenames = glob.glob(args.matrices_understand_dir + '/*' + commit_hash + '.csv')
        if len(understand_filenames) == 0:
            missing = True
            missing_understand.add(project_repo)
            logging.warning('Could not find understand matrix for %s' % project_repo)
        pmd_filenames = glob.glob(args.matrices_pmd_dir + '/*' + commit_hash + '.csv')
        if len(pmd_filenames) == 0:
            missing = True
            missing_pmd.add(project_repo)
            logging.warning('Could not find pmd matrix for %s' % project_repo)
        if missing:
            continue
        used_projects.add(project_repo)

        understand_frame = pd.read_csv(understand_filenames[0])
        understand_frame = understand_frame.drop('Kind', axis=1)
        # project_frame['Project'] = '-'.join(file_splitted)
        understand_frame['repository'] = project_repo
        understand_frame = understand_frame.set_index([
            'Name',
            'repository',
            # 'Project',
        ])
        # removes duplicates. TODO: why are there duplicates? Ask Cat (see, e.g., guava)
        understand_frame = understand_frame[~understand_frame.index.duplicated(keep='first')]

        for column_name in understand_frame.columns:
            if column_name != 'File':
                understand_frame[column_name] = understand_frame[column_name].astype(dtype=float)

        dimensions_old = project_code_smells.shape
        # the following line determines how to handle the records.
        # inner join means: only keep records that occur in both datasets
        joined_frame = project_code_smells.join(understand_frame, how='left')
        if joined_frame.shape[0] > dimensions_old[0]:
            raise ValueError('File %s Too much rows: %d vs %d' % (project_repo, joined_frame.shape[0], dimensions_old[0]))
        if joined_frame.shape[0] < dimensions_old[0]:
            raise ValueError('File %s: Expected %d rows after merge with understand, got only %d' % (project_repo, dimensions_old[0], len(joined_frame)))
        if joined_frame.shape[1] - understand_frame.shape[1] != 4:
            raise ValueError('File %s does not contain a plausible new column count. Old count %d, new count %d' % (project_repo, dimensions_old[1], joined_frame.shape[1], ))

        pmd_metrics = pd.read_csv(pmd_filenames[0])
        # fill nans with -999, then drop duplicated
        pmd_metrics = pmd_metrics[~pmd_metrics.fillna(-999).duplicated()]

        pmd_metrics['repository'] = project_repo
        pmd_metrics = pmd_metrics.set_index([
            'Name',
            'repository'
        ])

        # TODO: prevent duplicates
        dimensions_old = joined_frame.shape
        all_joined_frame = joined_frame.join(pmd_metrics, how='left')

        if len(all_joined_frame) < dimensions_old[0]:
            raise ValueError('File %s: Expected at least %d rows after merge with PMD, got only %d' % (project_repo, dimensions_old[0], all_joined_frame.shape[0]))
        elif len(all_joined_frame) > dimensions_old[0]:
            pmd_duplicate_rows += len(all_joined_frame) - dimensions_old[0]
            logging.warning('File %s: Expected at most %d rows after merge with PMD, got %d' % (project_repo, dimensions_old[0], all_joined_frame.shape[0]))
            # This happens when there are multiple PMD records per file
        if all_joined_frame.shape[1] != dimensions_old[1] + 6:
            raise ValueError('Before merge: %d columns, expected columns after merge: %d, actual: %d' % (dimensions_old[1], dimensions_old[1] + 6, all_joined_frame.shape[1]))

        # finally add to the list
        list_projects_frames.append(all_joined_frame)
        # some final checks
        if set(list_projects_frames[0].columns) != set(list_projects_frames[-1].columns):
            orig = set(list_projects_frames[0].columns)
            new = set(list_projects_frames[-1].columns)
            missing = orig - new
            additional = new - orig

            # happens.
            logging.warning('Column set does not match for %s. Missing: %s, Additional: %s' % (project_repo, missing, additional))

    output_file_csv = os.path.join(args.output_dir, "%s.csv" % args.smell_type)
    output_file_ignored = os.path.join(args.output_dir, "%s_ignored.json" % args.smell_type)
    output_file_included = os.path.join(args.output_dir, "%s_included.json" % args.smell_type)
    all_projects_frame = list_projects_frames[0].append(list_projects_frames[1:])
    all_projects_frame.to_csv(output_file_csv)
    with open(output_file_included, 'w') as fp:
        json.dump(list(included_projects), fp)
    with open(output_file_ignored, 'w') as fp:
        ignore_total_set = set(missing_pmd)
        ignore_total_set.update(missing_pmd)
        json.dump(list(ignore_total_set), fp)
    logging.info(all_projects_frame['severity'].value_counts())
    logging.info("data frame len %d" % len(all_projects_frame['severity']))
    logging.info("original frame len %d, pmd duplicates: %d" % (original_frame_len, pmd_duplicate_rows))
    logging.info("saved output file to: %s" % output_file_csv)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
