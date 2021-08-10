import argparse
import logging
import openml
import os
import pandas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blob', type=str, default=os.path.expanduser('~/experiments/code_smells/blob.csv'))
    parser.add_argument('--data_class', type=str, default=os.path.expanduser('~/experiments/code_smells/data class.csv'))
    parser.add_argument('--openml_apikey', type=str, default=None,
                        help='the openml api key')
    parser.add_argument('--openml_server', type=str, default=None,
                        help='the openml server location')

    return parser.parse_args()


def run(args):
    openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'

    blob = pandas.read_csv(args.blob)
    dataclass = pandas.read_csv(args.data_class)

    ignore_atts = ['repository', 'package', 'filename', 'code_name', 'commit_hash', 'smell', 'class_name']
    citation = 'Chitsutha Soomlek, Jan N. van Rijn and Marcello M. Bonsangue, Automatic human-like detection of code smells, Discovery Science 2021. '
    description = "This dataset combines records from the MLCQ dataset with metrics extracted using the PMD Tool and the Understand tool, to determine whether a file contains code smells. Please note that the records are on (sub)class level. Classification task, the default class (severity) should be binarized with a static threshold (preferably between 0.5 and 2.5). Please carefully read the publication to understand how to use this dataset. "
    blob = openml.datasets.create_dataset(
        name="Code_Smells_Blob",
        description=description,
        creator="Chitsutha Soomlek, Jan N. van Rijn and Marcello M. Bonsangue",
        contributor="MLCQ Team, PMD Team, Understand Team",
        collection_date="Spring 2021",
        language="English",
        licence=None,
        default_target_attribute="severity",
        row_id_attribute=None,
        ignore_attribute=ignore_atts,
        citation=citation,
        attributes="auto",
        data=blob,
        version_label="v1",
    )
    blob_id = blob.publish()
    dataclass = openml.datasets.create_dataset(
        name="Code_Smells_Data_Class",
        description=description,
        creator="Chitsutha Soomlek, Jan N. van Rijn and Marcello M. Bonsangue",
        contributor="MLCQ Team, PMD Team, Understand Team",
        collection_date="Spring 2021",
        language="English",
        licence=None,
        default_target_attribute="severity",
        row_id_attribute=None,
        ignore_attribute=ignore_atts,
        citation=citation,
        attributes="auto",
        data=dataclass,
        version_label="v1",
    )
    data_class_id = dataclass.publish()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
