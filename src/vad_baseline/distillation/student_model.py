"""Simplified CRDNN model for VAD distillation."""

import torch
import torch.nn as nn


class SimplifiedCRDNN(nn.Module):
    """
    A simplified CRDNN model for VAD.

    Architecture:
    - 2-layer CNN (channel reduction from original)
    - 1-layer GRU (smaller hidden size)
    - 1-layer DNN
    - Output projection to single value (speech probability)

    Target: < 0.5M parameters, < 1s inference time
    """

    def __init__(
        self,
        input_size: int = 257,  # fbank features
        cnn_channels: tuple = (32, 64),
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        dnn_hidden_size: int = 128,
    ):
        super().__init__()

        # CNN: (batch, time, freq) -> (batch, time, freq, 1) for conv2d
        self.conv1 = nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))  # Pool only frequency
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Compute CNN output size after pooling (2 pools with factor 2 each)
        cnn_out_size = cnn_channels[1] * (input_size // 4)

        # RNN: bidirectional GRU
        self.rnn = nn.GRU(
            input_size=cnn_out_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # Only 1 layer, no dropout
        )

        # DNN: project GRU output to hidden
        self.dnn = nn.Linear(rnn_hidden_size * 2, dnn_hidden_size)  # *2 for bidirectional
        self.dnn_activation = nn.ReLU()

        # Output: single speech probability
        self.output = nn.Linear(dnn_hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, time, freq) - fbank features

        Returns:
            (batch, time) - speech probabilities per frame
        """
        batch_size, time_steps, freq_bins = x.shape

        # CNN expects (batch, channel, time, freq)
        x = x.unsqueeze(1)  # (batch, 1, time, freq)

        # Conv layers with pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, 32, time, freq//2)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, 64, time, freq//4)

        # Reshape for RNN: (batch, time, features)
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq//4)
        x = x.reshape(batch_size, time_steps, -1)  # (batch, time, channels * freq//4)

        # RNN
        x, _ = self.rnn(x)  # (batch, time, rnn_hidden * 2)

        # DNN
        x = self.dnn(x)
        x = self.dnn_activation(x)
        x = self.dropout(x)

        # Output projection
        x = self.output(x)  # (batch, time, 1)

        # Squeeze and apply sigmoid for probability
        x = x.squeeze(-1)  # (batch, time)
        x = torch.sigmoid(x)

        return x