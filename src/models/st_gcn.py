"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) model
for skeleton-based action recognition
Based on: Skeleton-Based Action Recognition with Spatial-Temporal Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph:
    """
    Graph structure for human skeleton
    Using COCO keypoint format (17 keypoints)
    """
    
    def __init__(self, num_keypoints=17, layout='coco'):
        self.num_keypoints = num_keypoints
        self.layout = layout
        
        if layout == 'coco':
            # COCO 17 keypoint pairs (skeleton connections)
            self.edge = [
                # Head connections
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                # Torso
                [5, 11], [6, 12], [5, 6],
                # Left arm
                [5, 7], [7, 9],
                # Right arm
                [6, 8], [8, 10],
                # Left leg
                [11, 13], [13, 15],
                # Right leg
                [12, 14], [14, 16],
                # Additional connections for better graph connectivity
                [0, 1], [0, 2], [1, 3], [2, 4],  # Nose and eyes
                [0, 5], [0, 6]  # Head to shoulders
            ]
            
            # Center node (nose)
            self.center = 0
            
        self.A = self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self):
        """
        Build adjacency matrix for the graph
        Returns adjacency matrix A and normalized versions
        """
        num_node = self.num_keypoints
        
        # Initialize adjacency matrix
        A = np.zeros((num_node, num_node))
        
        # Build adjacency matrix from edges
        for edge in self.edge:
            A[edge[0], edge[1]] = 1
            A[edge[1], edge[0]] = 1  # Undirected graph
        
        # Add self-connections
        A = A + np.eye(num_node)
        
        # Normalize adjacency matrix
        D = np.sum(A, axis=1)  # Degree matrix
        D_inv = np.power(D, -0.5)
        D_inv[np.isinf(D_inv)] = 0
        D_inv = np.diag(D_inv)
        
        # Normalized adjacency matrix
        A_norm = D_inv @ A @ D_inv
        
        return torch.FloatTensor(A_norm)
    
    def get_adjacency_matrix(self):
        """Get normalized adjacency matrix"""
        return self.A


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer
    """
    
    def __init__(self, in_channels, out_channels, A, bias=True):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            A: Adjacency matrix
            bias: Whether to use bias
        """
        super(GraphConvolution, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, T, V, C_in)
        Returns:
            Output tensor of shape (batch, T, V, C_out)
        """
        batch_size, T, V, C_in = x.size()
        
        # Reshape: (batch*T, V, C_in)
        x = x.view(batch_size * T, V, C_in)
        
        # Graph convolution: X' = AXW
        # (batch*T, V, C_in) @ (C_in, C_out) -> (batch*T, V, C_out)
        x = torch.matmul(x, self.weight)
        
        # Apply adjacency matrix: (batch*T, V, C_out) @ (V, V) -> (batch*T, V, C_out)
        # Note: A is already normalized
        x = torch.matmul(self.A.to(x.device), x)
        
        # Add bias if applicable
        if self.bias is not None:
            x = x + self.bias
        
        # Reshape back: (batch, T, V, C_out)
        x = x.view(batch_size, T, V, self.out_channels)
        
        return x


class TemporalConvolution(nn.Module):
    """
    Temporal Convolutional Layer
    Applies 1D convolution along temporal dimension
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            kernel_size: Temporal convolution kernel size
            stride: Stride
        """
        super(TemporalConvolution, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(kernel_size // 2, 0)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, T, V, C_in)
        Returns:
            Output tensor of shape (batch, T', V, C_out)
        """
        # Permute: (batch, T, V, C) -> (batch, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Temporal convolution: (batch, C, T, V)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Permute back: (batch, C, T, V) -> (batch, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block
    Combines spatial (graph) and temporal convolutions
    """
    
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            A: Adjacency matrix
            stride: Temporal stride
            residual: Whether to use residual connection
        """
        super(STGCNBlock, self).__init__()
        
        self.residual = residual
        
        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Temporal convolution
        self.tcn = TemporalConvolution(out_channels, out_channels, kernel_size=9, stride=stride)
        
        # Residual connection
        if residual and (in_channels != out_channels or stride != 1):
            self.residual_conv = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=(1, 1),
                stride=(stride, 1)
            )
            self.residual_bn = nn.BatchNorm2d(out_channels)
        else:
            self.residual_conv = None
            self.residual_bn = None
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, T, V, C_in)
        Returns:
            Output tensor of shape (batch, T', V, C_out)
        """
        residual = x
        
        # Spatial convolution
        x = self.gcn(x)
        
        # Batch norm: (batch, T, V, C) -> (batch, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.bn1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.relu(x)
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Residual connection
        if self.residual:
            if self.residual_conv is not None:
                residual = residual.permute(0, 3, 1, 2)
                residual = self.residual_conv(residual)
                residual = self.residual_bn(residual)
                residual = residual.permute(0, 2, 3, 1)
            x = x + residual
        
        return x


class STGCN(nn.Module):
    """
    ST-GCN Model for skeleton-based action recognition
    """
    
    def __init__(self, num_classes=101, in_channels=3, graph_layout='coco',
                 num_stages=4, hidden_channels=[64, 64, 128, 256], 
                 dropout=0.5):
        """
        Args:
            num_classes: Number of action classes
            in_channels: Input channels per keypoint (x, y, score = 3)
            graph_layout: Graph layout type ('coco')
            num_stages: Number of ST-GCN stages
            hidden_channels: List of hidden channel sizes for each stage
            dropout: Dropout rate
        """
        super(STGCN, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.hidden_channels = hidden_channels
        
        # Build graph
        self.graph = Graph(num_keypoints=17, layout=graph_layout)
        A = self.graph.get_adjacency_matrix()
        
        # Input layer
        self.input_gcn = GraphConvolution(in_channels, hidden_channels[0], A)
        self.input_bn = nn.BatchNorm2d(hidden_channels[0])
        self.input_relu = nn.ReLU()
        
        # ST-GCN blocks
        self.stgcn_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        
        for i in range(num_stages):
            out_ch = hidden_channels[min(i, len(hidden_channels) - 1)]
            stride = 2 if i > 0 else 1  # Downsample after first stage
            self.stgcn_blocks.append(
                STGCNBlock(in_ch, out_ch, A, stride=stride, residual=True)
            )
            in_ch = out_ch
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch, T, V, C)
               where T=time, V=keypoints, C=coordinates
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Input layer
        x = self.input_gcn(x)
        # Batch norm: (batch, T, V, C) -> (batch, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.input_bn(x)
        x = x.permute(0, 2, 3, 1)
        x = self.input_relu(x)
        
        # ST-GCN blocks
        for block in self.stgcn_blocks:
            x = block(x)
        
        # Global pooling
        # (batch, T, V, C) -> (batch, C, T, V) -> (batch, C, 1, 1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'STGCN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'num_stages': self.num_stages,
            'hidden_channels': self.hidden_channels
        }


if __name__ == '__main__':
    # Test model
    model = STGCN(num_classes=101, in_channels=3, 
                  hidden_channels=[64, 64, 128, 256], dropout=0.5)
    
    # Test input
    batch_size = 8
    T, V, C = 300, 17, 3
    x = torch.randn(batch_size, T, V, C)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

