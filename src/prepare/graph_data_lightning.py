from src.prepare import DatasetBuilder, JoernExeError

import os
import pytorch_lightning as pl
import torch
import torch.utils.data as data_utils
import torch_geometric.data as geom_data
import torch_geometric.transforms as geom_transforms

class GraphDataLightning(pl.LightningDataModule):
    """ Lightning Data Module for graph datasets. Batch size 128.
    """
    def __init__(self,
                 fresh_build: bool,       
                 test_build: bool, 
                 num_nodes: int,
                 train_proportion: float,
                 batch_size: int) -> None:
        """
        Args:
            fresh_build: True to geenrate dataset source files and graphs,
                deleting any existing. False to reuse existing files.
            test_build: True to use a subset of the data 
                (from the project directory's test folder), 
                False if all data is to be used.
            num_nodes: The number of nodes to be used in an individual CPG.
            train_proportion: Proportion of samples used for the training dataset.
                The rest is split in half for test/val.
            batch_size

        Raises:
            JoernExeError: If running Joern fails
        """
        super().__init__()
        try:
            dataset = DatasetBuilder(fresh_build, test_build).build_graphs(num_nodes)
        except JoernExeError:
            raise JoernExeError    
        train_count = int(len(dataset) * train_proportion)
        holdout_count = len(dataset) - train_count
        test_count = int(holdout_count/2)
        val_count = holdout_count - test_count
        split_counts = [train_count, test_count, val_count]
        self.batch_size = batch_size
        self.train_set, self.test_set, self.val_set = data_utils.random_split(dataset, split_counts)

    def train_dataloader(self) -> geom_data.DataLoader:
        negative_sample_cnt = len([x for x in self.train_set if x.y == 0])
        positive_sample_cnt = len([x for x in self.train_set if x.y == 1])
        # more negative samples than positive in this dataset
        oversample_rate = negative_sample_cnt / positive_sample_cnt
        class_weights = [1, oversample_rate]
        sample_weights = [0] * len(self.train_set)
        for index, (_, _, y) in enumerate(self.train_set):
            label = y[1][0].item()
            class_weight = class_weights[label]
            sample_weights[index] = class_weight
        sampler = data_utils.WeightedRandomSampler(sample_weights, 
                                                   num_samples=len(self.train_set))
        loader = geom_data.DataLoader(self.train_set, 
                                      batch_size = self.batch_size,
                                      num_workers = os.cpu_count(),
                                      sampler = sampler)
        return loader
    
    def test_dataloader(self) -> geom_data.DataLoader:
        loader = geom_data.DataLoader(self.test_set,
                                      batch_size= self.batch_size,
                                      num_workers=os.cpu_count())
        return loader

    def val_dataloader(self) -> geom_data.DataLoader:
        loader = geom_data.DataLoader(self.val_set,
                                      batch_size= self.batch_size,
                                      num_workers=os.cpu_count())
        return loader
