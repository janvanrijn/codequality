import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/Downloads/MLCQ_software_matrices/'))
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

    all_projects_frame = None
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
        project_frame = project_frame.join(project_code_smells, how='left')

        if all_projects_frame is None:
            all_projects_frame = project_frame
        else:
            all_projects_frame = all_projects_frame.append(project_frame)
            logging.info("all_projects_frame: %s" % str(all_projects_frame.shape))
    all_projects_frame.to_csv(os.path.join(args.output_dir, "%s.csv" % args.smell_type))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
