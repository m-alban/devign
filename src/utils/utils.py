import argparse
import os
import pathlib


PROJECT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    top_subparser = parser.add_subparsers(help = 'Setup or run tasks.')
    top_subparser.add_parser('setup')
        #'setup',
        #help = 'Setup directories for data and embeddings.',
        #action = 'store_const',
        #const = 'setup'
    #)
    run_parser = top_subparser.add_parser('run')
    #    'run',
    #    help = 'Run data prep or a model.',
    #    action = 'store_const',
    #    const = 'run'
    #)
    run_parser.add_argument(
        'scope',
        help = 'Whether to use the full dataset or a small subset.',
        choices = ['sample', 'full']
    )
    run_subparsers = run_parser.add_subparsers(help = 'Prepare data or run model.')
    run_subparsers.add_parser('prepare')
    model_parser = run_subparsers.add_parser('model')
    model_parser.add_argument('architecture',
                              help = 'Architecture of the network.',
                              choices = ['flat', 'devign'])
    model_parser.add_argument('--rebuild',
                              help = 'Flag to also prepare the data',
                              #metavar = '',
                              action = 'store_true',
                              required=False)
    """
    parser.add_argument('scope',
                         help = 'Whether to use the full dataset or a small subset.',
                         choices = ['sample', 'full'])
    subparsers = parser.add_subparsers(help = 'Prepare data or train (and test) model.')
    model_parser = subparsers.add_parser('model')
    model_parser.add_argument('architecture',
                              help = 'Architecture of the network.',
                              choices = ['flat', 'devign'])
    model_parser.add_argument('--rebuild',
                              help = 'Flag to also prepare the data',
                              metavar = '',
                              action = 'store_true',
                              required=False)
    prepare_parser = subparsers.add_parser('prepare')
    """
    return parser.parse_args()

def process_joern_error(test_build: bool) -> int:
    if test_build:
        src_file_dir_path = ['test']
    else:
        src_file_dir_path = []
    src_file_dir_path.extend(['data', 'src_files'])
    src_file_dir = os.path.join(PROJECT_ROOT, *src_file_dir_path)
    err_message = (
        'Joern failed, try running directly: \n'
        '</path/to/joern_executable> --script '
        f'{PROJECT_ROOT}/joern/export_cpg.sc '
        f'--params src_file_dir="{src_file_dir}"'
    )
    print(err_message)
    return 1

def setup() -> None:
    scope_dirs = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'sample')]
    for scope_path in scope_dirs:
        sample_path_used = 'sample' in str(scope_path)
        if sample_path_used:
            err_message = 'Sample'
            err_dir_path = './sample'
        else:
            err_message = 'Full'   
            err_dir_path = './'
        # model directory for embeddings
        model_dir = os.path.join(scope_path, 'models')
        try:
            os.mkdir(model_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'models')
            print(f'{err_message} dataset embedding folder already exists:')
            print(f'    {err_path_message}')
        # data folders
        data_folder = os.path.join(scope_path, 'data')
        err_dir_path = os.path.join(err_dir_path, 'data')
        graphs_dir = os.path.join(data_folder, 'graphs')
        src_file_dir = os.path.join(data_folder, 'src_files')
        try:
            os.mkdir(graphs_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'graphs')
            print(f'{err_message} dataset graphs folder already exists:')
            print(f'    {err_path_message}')
        try:
            os.mkdir(src_file_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'src_files')
            print(f'{err_message} source files folder already exists:')
            print(f'    {err_path_message}')