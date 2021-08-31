from src.model import GGNNFlatSum, Devign
from src.metrics import BinarySensitivity

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch_geometric.data as geom_data
import pytorch_lightning as pl

architecture_dict = {'devign': Devign,
                     'flat': GGNNFlatSum}


class DevignLightning(pl.LightningModule):
    """ Lightning module for training. Optimized with AdamW
    """
    def __init__(self, architecture_name: str, lr: float, **model_kwargs) -> None:
        """
        Args:
            architecture_name - Name of network architecture, key
                from [architecture_dict]
            model_kwargs - Additional arguments for the network
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = architecture_dict[architecture_name](**model_kwargs)
        self.loss_module = nn.functional.binary_cross_entropy
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=2),
            torchmetrics.F1(multiclass=False),
            torchmetrics.Specificity(multiclass=False),
            BinarySensitivity()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, data: geom_data.batch.Batch) -> torch.Tensor:
        x = self.model(data.x, data.edge_index, data.batch)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, 
                      batch: geom_data.batch.Batch,
                      batch_index: int) -> torch.Tensor:
        out = self.forward(batch)
        y = batch.y
        loss = self.loss_module(out, y.float())
        preds = (out > 0.5).int()
        self.log('loss', loss)
        metric_out = self.train_metrics(preds, y)
        self.log_dict(metric_out, on_step=True)
        return loss
    
    def validation_step(self, 
                       batch: geom_data.batch.Batch,
                       batch_index: int) -> torch.Tensor:
        out = self.forward(batch)
        preds = (out > 0.5).int()
        metric_out = self.val_metrics(preds, batch.y)
        self.log_dict(metric_out)

    def test_step(self, 
                  batch: geom_data.batch.Batch,
                  batch_index: int) -> torch.Tensor:
        out = self.forward(batch)
        preds = (out > 0.5).int()
        metric_out = self.test_metrics(preds, batch.y)
        self.log_dict(metric_out)
    
