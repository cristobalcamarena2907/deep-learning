"""
Baseline LSTM model for skeleton-based action recognition
Simple LSTM that processes keypoint sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLSTM(nn.Module):
    """
    Baseline LSTM model for action recognition
    Processes skeleton sequences using LSTM layers
    """
    
    def __init__(self, num_classes=101, input_dim=3, hidden_dim=128, 
                 num_layers=2, dropout=0.5, bidirectional=True):
        """
        Args:
            num_classes: Number of action classes
            input_dim: Input dimension per keypoint (x, y, score = 3)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BaselineLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Number of keypoints in COCO format
        self.num_keypoints = 17
        
        # Flatten keypoints: (T, V, C) -> (T, V*C)
        # Input to LSTM will be flattened keypoints per frame
        lstm_input_dim = self.num_keypoints * input_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, T, V, C)
               where T=time, V=keypoints, C=coordinates
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        T, V, C = x.size(1), x.size(2), x.size(3)
        
        # Flatten keypoints: (batch, T, V, C) -> (batch, T, V*C)
        x = x.view(batch_size, T, V * C)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            forward_hidden = h_n[-2]  # Last forward hidden state
            backward_hidden = h_n[-1]  # Last backward hidden state
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = h_n[-1]
        
        # Alternative: Use last output instead of hidden state
        # final_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Fully connected layers
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'BaselineLSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }


if __name__ == '__main__':
    # Test model
    model = BaselineLSTM(num_classes=101, input_dim=3, hidden_dim=128, 
                        num_layers=2, dropout=0.5)
    
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

