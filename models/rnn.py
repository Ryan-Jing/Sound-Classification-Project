import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=10, dropout=0.5):
        """
        A simple RNN model using LSTM for audio classification.

        Args:
            input_size (int): The number of features in the input (e.g., number of Mel bands).
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): The number of output classes.
            dropout (float): Dropout probability.
        """
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Use a bidirectional LSTM
        )
        
        # The linear layer will take the concatenated hidden states from both directions
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x is expected to be of shape (batch, channels, n_mels, time)
        # We need to reshape it for the LSTM: (batch, time, n_mels)
        # Squeeze the channel dimension and permute
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        # LSTM forward pass
        # h0 and c0 are initialized to zero by default
        out, _ = self.lstm(x)
        
        # We take the output from the last time step
        # out is of shape (batch, seq_len, hidden_size * 2)
        last_hidden_state = out[:, -1, :]
        
        # Apply dropout and the final fully connected layer
        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        
        return out
