import argparse
import json
import logging
import os
import pandas as pd
import re


def parse_args():
    parser = argparse.ArgumentParser('transforms PMD matrices')
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/data/codequality/PMD_matrices_raw/'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/data/codequality/PMD_matrices_processed/'))
    parser.add_argument('--max_projects', type=int, default=None)

    return parser.parse_args()


def run(args):
    files = os.listdir(args.matrices_dir)
    if args.max_projects:
        files = files[:args.max_projects]
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, file in enumerate(files):
        file_extension = os.path.splitext(file)[-1]
        if os.path.splitext(file)[-1] != '.csv':
            logging.info("skipping file: %s (extension %s)" % (file, file_extension))
            continue

        all_data = []
        filename_base = os.path.splitext(file)[0]
        project_name = '-'.join(filename_base.split('-')[:-1])
        commitHash = filename_base.split('-')[-1]
        project_frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        dims = project_frame.shape
        logging.info("opened file: %s, shape: %s (%d/%d)" % (file, str(dims), idx + 1, len(files)))
        project_frame = project_frame.drop_duplicates()
        # if project_frame.shape != dims:
        #     logging.warning('Removed duplicates, new dimensions: (%d,%d)' % project_frame.shape)

        for _, row in project_frame.iterrows():
            filename = os.path.basename(row['File'])
            main_class_name = os.path.splitext(filename)[0]
            # obtains the classname of the class that has code smell
            smell_class_name = row['Description'][len('Java matrics of class '):].split(' ')[0]
            # obtains all text between brackets
            metrics_unparsed = re.search(r'\((.*?)\)', row['Description']).group(1)
            # removes procent signs and illegal numbers
            metrics_unparsed = metrics_unparsed.replace('%', '').replace('NAN', '')
            # prepares the string to be json parsable
            metrics_unparsed = '{"' + metrics_unparsed.replace('=', '": "').replace(', ', '","') + '"}'
            # TODO: parse away the , between 1000 numbers!!
            record = json.loads(metrics_unparsed)
            for key, item in record.items():
                if ',' in item:
                    # logging.warning('file %s contains numbers with comma in it: %s. Will be replaced.' % (file, item))
                    record[key] = item.replace(',', '')
            record['package'] = row['Package']
            record['filename'] = filename
            record['class_name'] = smell_class_name
            all_data.append(record)
        if len(all_data) == 0:
            logging.warning('Project %s (%s) does not contain any matrices' % (project_name, commitHash))
            continue

        result = pd.DataFrame(all_data)
        result = result.set_index(['package', 'filename'])
        result.to_csv(os.path.join(args.output_dir, '%s-%s.csv' % (project_name, commitHash)))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
