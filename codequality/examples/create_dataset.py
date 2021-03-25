import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/data/codequality/MLCQ_matrices'))
    parser.add_argument('--matrices_more_dir', type=str, default=os.path.expanduser('~/data/codequality/PMD_matrices_processed'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/experiments/code_smells/'))
    parser.add_argument('--code_smells_path', type=str, default=os.path.expanduser('../data/code_smells.csv'))
    parser.add_argument('--max_projects', type=int, default=None)
    parser.add_argument('--smell_type', type=str, default='data class')

    return parser.parse_args()


def run(args):
    files = os.listdir(args.matrices_dir)
    if args.max_projects:
        files = files[:args.max_projects]
    os.makedirs(args.output_dir, exist_ok=True)
    missing_matrices = 0

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
    all_code_smells_frame['CommitHash'] = all_code_smells_frame['commit_hash']
    all_code_smells_frame['Name'] = all_code_smells_frame['code_name']
    all_code_smells_frame = all_code_smells_frame[['CommitHash', 'Name', 'smell', 'severity']]
    all_code_smells_frame = all_code_smells_frame.groupby(['CommitHash', 'Name', 'smell']).mean()
    all_code_smells_frame = all_code_smells_frame.reset_index()
    logging.info("Number of records: %d" % len(all_code_smells_frame))

    list_projects_frames = []
    for idx, file in enumerate(files):
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue
        logging.info("opening file: %s (%d/%d)" % (file, idx+1, len(files)))
        file_prefix = os.path.splitext(file)[0]
        file_splitted = file_prefix.split('-')

        project_code_smells = all_code_smells_frame[all_code_smells_frame['CommitHash'] == file_splitted[-1]]
        project_code_smells = project_code_smells.set_index(['CommitHash', 'Name'])

        logging.info("code smells: %d" % len(project_code_smells))

        project_frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        project_frame = project_frame.drop('Kind', axis=1)
        # project_frame['Project'] = '-'.join(file_splitted)
        # commit hash is actually overkill..
        project_frame['CommitHash'] = file_splitted[-1]
        project_frame = project_frame.set_index([
            'CommitHash',
            'Name',
            # 'Project',
        ])
        # removes duplicates. TODO: why are there duplicates? Ask Cat
        project_frame = project_frame[~project_frame.index.duplicated(keep='first')]

        # TODO: catch if not exists (need to be away for numeric frame)
        if 'File' in project_frame.columns:
            project_frame = project_frame.drop('File', axis=1)
        else:
            logging.warning('File %s does not contain a "file" column' % file)

        project_frame = project_frame.astype(dtype=float)
        dimensions_old = project_frame.shape
        # the following line determines how to handle the records.
        # inner join means: only keep records that occur in both datasets
        project_frame = project_frame.join(project_code_smells, how='inner')
        if project_frame.shape[0] > len(project_code_smells):
            raise ValueError('File %s Too much rows: %d vs %d' % (file, project_frame.shape[0], len(project_code_smells)))
        if project_frame.shape[0] < len(project_code_smells):
            missing_matrices += len(project_code_smells) - project_frame.shape[0]
            logging.warning('File %s: Expected %d rows after merge with MLCQ, got only %d' % (file, len(project_code_smells), project_frame.shape[0], ))
        if project_frame.shape[1] - dimensions_old[1] != 2:
            raise ValueError('File %s does not contain a plausible new column count' % file)

        more_metrics_file = os.path.join(args.matrices_more_dir, file)
        more_metrics = pd.read_csv(more_metrics_file)
        # prevent mix-up with scientific notation
        more_metrics['CommitHash'] = more_metrics['CommitHash'].astype(str)
        more_metrics = more_metrics.set_index([
            'CommitHash',
            'Name'
        ])
        more_metrics = more_metrics.astype(dtype=float)

        # TODO: prevent duplicates
        dimensions_old = project_frame.shape
        project_frame = project_frame.join(more_metrics, how='inner')
        list_projects_frames.append(project_frame)
        if set(list_projects_frames[0].columns) != set(list_projects_frames[-1].columns):
            orig = set(list_projects_frames[0].columns)
            new = set(list_projects_frames[-1].columns)
            missing = orig - new
            additional = new - orig

            # happens.
            logging.warning('Column set does not match for %s. Missing: %s, Additional: %s' % (file, missing, additional))
            continue
        if dimensions_old[0] != len(project_frame):
            logging.warning('File %s: Expected %d rows after merge with PMD, got only %d' % (file, dimensions_old[0], project_frame.shape[0]))

    all_projects_frame = list_projects_frames[0].append(list_projects_frames[1:])
    all_projects_frame.to_csv(os.path.join(args.output_dir, "%s.csv" % args.smell_type))
    print(all_projects_frame['severity'].value_counts())
    print("data frame len %d" % len(all_projects_frame['severity']))
    print("missing matrices count: %d" % missing_matrices)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
