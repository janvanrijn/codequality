import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/Downloads/MLCQ_software_matrices/'))
    parser.add_argument('--matrices_more_dir', type=str, default=os.path.expanduser('~/Downloads/matrices_processed/'))
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

    all_code_smells_frame = pd.read_csv(args.code_smells_path)
    all_code_smells_frame = all_code_smells_frame[all_code_smells_frame['smell'] == args.smell_type]
    all_code_smells_frame['CommitHashPrefix'] = all_code_smells_frame['commit_hash'].str[:7]
    all_code_smells_frame['Name'] = all_code_smells_frame['code_name']
    all_code_smells_frame = all_code_smells_frame[['CommitHashPrefix', 'Name', 'smell', 'severity']]

    list_projects_frames = []
    for idx, file in enumerate(files):
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue
        logging.info("opening file: %s (%d/%d)" % (file, idx+1, len(files)))
        file_prefix = os.path.splitext(file)[0]
        file_splitted = file_prefix.split('-')

        project_code_smells = all_code_smells_frame[all_code_smells_frame['CommitHashPrefix'] == file_splitted[-1]]
        project_code_smells = project_code_smells.set_index(['CommitHashPrefix', 'Name'])

        logging.info("code smells: %d" % len(project_code_smells))

        project_frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        project_frame = project_frame[project_frame['Kind'] == 'Public Class'].drop('Kind', axis=1)
        # project_frame['Project'] = '-'.join(file_splitted)
        project_frame['CommitHashPrefix'] = file_splitted[-1]
        project_frame = project_frame.set_index([
            'CommitHashPrefix',
            'Name',
            # 'Project',
        ])
        # TODO: catch if not exists (need to be away for numeric frame)
        if 'File' in project_frame.columns:
            project_frame = project_frame.drop('File', axis=1)
        else:
            logging.warning('File %s does not contain a "file" column' % file)

        project_frame = project_frame.astype(dtype=float)
        dimensions_old = project_frame.shape
        project_frame = project_frame.join(project_code_smells, how='left')
        dimensions_new = project_frame.shape
        if dimensions_new[0] != dimensions_old[0]:
            raise ValueError('File %s New Rows have been introduced: %d vs %d' % (file, dimensions_new[0], dimensions_old[0]))
        if dimensions_new[1] - dimensions_old[1] != 2:
            raise ValueError('File %s does not contain a plausible new column count' % file)

        more_metrics_file = os.path.join(args.matrices_more_dir, file)
        if not os.path.isfile(more_metrics_file):
            logging.warning('Could not find more metrics for %s' % file)
            continue

        more_metrics = pd.read_csv(more_metrics_file)
        # prevent mixup with scientific notation
        more_metrics['CommitHashPrefix'] = more_metrics['CommitHashPrefix'].astype(str)
        more_metrics = more_metrics.set_index([
            'CommitHashPrefix',
            'Name'
        ])
        more_metrics = more_metrics.astype(dtype=float)
        if len(more_metrics) != len(project_frame):
            logging.warning('%s: Project frame contains %d lines, new metrics contains %d' % (file, len(project_frame), len(more_metrics)))

        project_frame = project_frame.join(more_metrics, how='left')
        list_projects_frames.append(project_frame)
        if set(list_projects_frames[0].columns) != set(list_projects_frames[-1].columns):
            orig = set(list_projects_frames[0].columns)
            new = set(list_projects_frames[-1].columns)
            missing = orig - new
            additional = new - orig

            # happens.
            logging.warning('Column set does not match for %s. Missing: %s, Additional: %s' % (file, missing, additional))
            continue

    all_projects_frame = list_projects_frames[0].append(list_projects_frames[1:])
    all_projects_frame.to_csv(os.path.join(args.output_dir, "%s.csv" % args.smell_type))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
