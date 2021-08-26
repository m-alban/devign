import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch_geometric.transforms as geom_transforms

from typing import Union

class GGNNFlatSum(nn.Module):
    """ Graph level prediction with gated graph recurrent networks
    """

    def __init__(self, 
                 input_channels: int, 
                 hidden_channels: int, 
                 num_layers:int, 
                 aggr: str = 'add', 
                 bias: bool = True, 
                 **kwargs) -> None:
        """
        Args:
            input_channels - Dimension of the input features
            hidden_channels - Dimension of hidden features
            num_layers - Number of layers for the Recurrent Graph Network
            aggr - Aggregation method used for the recurrent graph network
            bias - If set to false, the recurrent network will not learn an additive bias
            kwargs - Additional arguments for the GatedGraphConv model
        """
        super().__init__()
        self.ggnn = geom_nn.GatedGraphConv(out_channels=hidden_channels,
                                           num_layers=num_layers, aggr=aggr,
                                           bias=bias, **kwargs)
        # MLP
        self.head = nn.Sequential(
                nn.Dropout(),
                nn.Linear(input_channels + hidden_channels, 1),
        )

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges
                in the graph (Pytorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        out = self.ggnn(x=x, edge_index=edge_index)
        x = torch.cat((x, out), axis=1)
        x = self.head(x)
        x = geom_nn.global_add_pool(x, batch_index)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x

class Devign(nn.Module):
    """ Implementation of Devign
    """

    def __init__(self, 
                 input_channels: int, 
                 hidden_channels: int, 
                 num_layers: int, 
                 aggr: str= 'add', 
                 bias: bool= True, 
                 **kwargs) -> None:
        """
        Args:
            input_channels - Dimension of the input features
            hidden_channels - Dimension of hidden features
            num_layers - Number of layers for the Recurrent Graph Network
            aggr - Aggregation method used for the recurrent graph network
            bias - If set to false, the recurrent network will not learn an additive bias
            kwargs - Additional arguments for the GatedGraphConv model
        """
        super().__init__()
        self.ggnn = geom_nn.GatedGraphConv(out_channels=hidden_channels,
                                           num_layers=num_layers,
                                           aggr=aggr, **kwargs)
        # Wide components used for concatenation of ggnn output and original feature vectors
        self.relu = nn.ReLU
        wide_dimension = hidden_channels + input_channels
        conv_layers_wide = [
            nn.Conv1d(in_channels=200, 
                      out_channels=50,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(in_channels=50, 
                      out_channels=1,
                      kernel_size=1),
            self.relu(),
            nn.MaxPool1d(2, stride=2)
        ]
        self.conv_wide = nn.ModuleList(conv_layers_wide)
        # find dimension of input to MLP
        wide_out_dim = wide_dimension
        for layer in conv_layers_wide:
            if not isinstance(layer, self.relu):
                wide_out_dim = self.conv_layer_out_dim(wide_out_dim, layer)
        self.mlp_wide = nn.Sequential(
            nn.Dropout(),
            nn.Linear(wide_out_dim, 1),
        )
        # Narrow components used just for the output of the ggnn
        conv_layers_narrow = [
            nn.Conv1d(in_channels=200, 
                      out_channels=50,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(in_channels=50, 
                      out_channels=1,
                      kernel_size=1),
            self.relu(),
            nn.MaxPool1d(2, stride=2),
        ]
        self.conv_narrow = nn.ModuleList(conv_layers_narrow)
        # find dimension of input to MLP
        narrow_out_dim = hidden_channels
        for layer in conv_layers_narrow:
            if not isinstance(layer, self.relu):
                narrow_out_dim = self.conv_layer_out_dim(narrow_out_dim, layer)
        self.mlp_narrow = nn.Sequential(
            nn.Dropout(),
            nn.Linear(narrow_out_dim, 1),
        )
        
    def conv_layer_out_dim(self, 
                           input_dim: int, 
                           layer: Union[nn.Conv1d, nn.MaxPool1d]) -> int:
        """Compute output dimension of convolutional layer (or maxpool)"""
        layer_params = [layer.kernel_size, layer.padding, layer.stride]
        if isinstance(layer, nn.Conv1d):
            layer_params = [p[0] for p in layer_params]
        kernel, padding, stride = layer_params
        out_dim = (((input_dim + 2*padding - kernel)/stride)+1)
        return int(out_dim)

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_index: torch.Tensor) -> torch.Tensor:
        recurrent_out = self.ggnn(x=x, edge_index=edge_index)
        z = torch.cat((x, recurrent_out), axis=1)
        z_node_dim = z.shape[1]
        # resize to batch_size, num_nodes, node_dim + ggrn_out_dim
        z = z.view(-1, 200, z_node_dim)
        for l in self.conv_wide:
            z = l(z)
        # remove dimension holding channels for conv
        z = z.squeeze(1)
        z = self.mlp_wide(z)
        # squeeze from [batch_size, 1] to tensor len = batch_size
        z = z.squeeze(1)
        y = recurrent_out
        y_node_dim = y.shape[1]
        # resize to batch_size, num_nodes, ggrn_out_dim
        y = y.view(-1, 200, y_node_dim)
        for l in self.conv_narrow:
            y = l(y)
        # remove dimension holding channels for conv
        y = y.squeeze(1)
        y = self.mlp_narrow(y)
        # squeeze from [batch_size, 1] to tensor len = batch_size
        y = y.squeeze(1)
        out = torch.mul(z, y)
        out = torch.sigmoid(out)
        return out 
