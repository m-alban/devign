import argparse
import torch
import numpy as np
import os
import pytorch_lightning as pl

from src.model import DevignLightning 
from src.prepare import JoernExeError, GraphDataLightning, DatasetBuilder
from src.utils import PROJECT_ROOT

def main(args):
    test_build = args.scope == 'sample'
    if 'architecture' not in args:
        # prepare was invoked
        try:
            DatasetBuilder(fresh_build=True, test_build=test_build)
            return 0
        except JoernExeError:
            return process_joern_error(test_build)
    # model run was invoked
    model_kwargs = {
        'input_channels': 115,
        'hidden_channels': 200,
        'num_layers': 6
    }
    lr = 1e-4
    data_kwargs = {
        'fresh_build': args.rebuild, 
        'test_build': test_build,
        'num_nodes': np.inf if args.architecture == 'flat' else 200,
        'train_proportion': 0.70,
        'batch_size': 128,
    }
    pl.seed_everything(42)
    model = DevignLightning(args.architecture, lr, **model_kwargs)
    try:
        data_module = GraphDataLightning(**data_kwargs)
    except JoernExeError:
        return process_joern_error(test_build)
    trainer = pl.Trainer(gpus= 1 if torch.cuda.is_available() else 0,
                         max_epochs= 1,# if test_build else 200
                         log_every_n_steps = 10 if data_kwargs['test_build'] else 50)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)
    train_results = trainer.test(dataloaders = train_dataloader, verbose=False)
    print('TRAIN RESULTS')
    for metric, value in train_results[0].items():
        print('  ', metric.replace('test', 'train'))
        print('    ', value)
    test_results = trainer.test(dataloaders = test_dataloader, verbose=False)
    print('TEST RESULTS')
    for metric, value in test_results[0].items():
        print('  ', metric)
        print('    ', value)
    return 0

def parse_args():
    parser = argparse.ArgumentParser()
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
                              action = 'store_const',
                              const = True,
                              required=False)
    prepare_parser = subparsers.add_parser('prepare')
    return parser.parse_args()

def process_joern_error(test_build):
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

if __name__ == "__main__":
    args = parse_args()
    main(args)
