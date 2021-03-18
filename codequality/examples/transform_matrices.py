import argparse
import json
import logging
import os
import pandas as pd
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices_dir', type=str, default=os.path.expanduser('~/Downloads/matrices/'))
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~/Downloads/matrices_processed/'))
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
        logging.info("opening file: %s (%d/%d)" % (file, idx + 1, len(files)))

        all_data = []
        filename_base = os.path.splitext(file)[0]
        project_name = '-'.join(filename_base.split('-')[:-1])
        commitHash = filename_base.split('-')[-1]
        commitHashPrefix = commitHash[:7]
        project_frame = pd.read_csv(os.path.join(args.matrices_dir, file))
        for _, row in project_frame.iterrows():
            basename = os.path.basename(row['File'])
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
                    logging.warning('file %s contains numbers with comma in it: %s. Will be replaced.' % (file, item))
                    record[key] = item.replace(',', '')
            record['CommitHashPrefix'] = commitHashPrefix
            record['Name'] = "%s.%s" % (row['Package'], os.path.splitext(basename)[0])
            all_data.append(record)
        if len(all_data) == 0:
            logging.warning('Project %s (%s) does not contain any matrices' % (project_name, commitHashPrefix))
            continue

        result = pd.DataFrame(all_data)
        # ensure commithas is a string, so it does not get mistaken for scientific notation
        result['CommitHashPrefix'] = result['CommitHashPrefix'].astype(str)
        result = result.set_index(['CommitHashPrefix', 'Name'])
        result.to_csv(os.path.join(args.output_dir, '%s-%s.csv' % (project_name, commitHashPrefix)))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
