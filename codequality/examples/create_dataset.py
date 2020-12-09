import argparse
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/Downloads/MLCQ_software_matrices/'))
    parser.add_argument('--code_smells_path', type=str, default=os.path.expanduser('../data/code_smells.csv'))
    parser.add_argument('--max_projects', type=int, default=50)

    return parser.parse_args()


def run(args):
    files = os.listdir(args.matrices_dir)
    if args.max_projects:
        files = files[:args.max_projects]

    code_smells_frame = pd.read_csv(args.code_smells_path)

    all_projects_frame = None
    for idx, file in enumerate(files):
        logging.info("opening file: %s (%d/%d)" % (file, idx+1, len(files)))
        file_prefix = os.path.splitext(file)[0]
        file_splitted = file_prefix.split('-')

        project_frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        project_frame = project_frame[project_frame['Kind'] == 'Public Class'].drop('Kind', axis=1)
        project_frame['Project'] = '-'.join(file_splitted)
        project_frame['CommitHashPrefix'] = file_splitted[-1]
        project_frame = project_frame.set_index(['Name', 'Project', 'CommitHashPrefix'])
        if all_projects_frame is None:
            all_projects_frame = project_frame
        else:
            all_projects_frame = all_projects_frame.append(project_frame)
            logging.info("all_projects_frame: %s" % str(all_projects_frame.shape))



if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
