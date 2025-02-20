import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, N_INPUT: int, N_OUTPUT: int, N_HIDDEN: int, N_LAYERS: int):
        super().__init__()

        activation = nn.Tanh  # You can switch to ReLU, etc.

        # Input layer
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )

        # Hidden layers
        if N_LAYERS > 1:
            hidden_layers = []
            for _ in range(N_LAYERS - 1):
                hidden_layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
                hidden_layers.append(activation())
            self.fch = nn.Sequential(*hidden_layers)
        else:
            self.fch = nn.Identity()  # No hidden layers

        # Output layer
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)  # Input layer
        x = self.fch(x)  # Hidden layers
        x = self.fce(x)  # Output layer
        return x
class TimeRNN(nn.Module):
    """
    Simple RNN or LSTM-based model that processes a sequence of times
    (shape: [batch, seq_len, 1]) and outputs a sequence of y(t) predictions 
    (shape: [batch, seq_len, 1]).
    """
    def __init__(self, hidden_size=32, rnn_type="LSTM"):
        super().__init__()
        self.hidden_size = hidden_size
        
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                               num_layers=1, batch_first=True)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, 
                              num_layers=1, batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'RNN'.")
        
        # Final linear layer to map hidden state -> scalar y(t)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: shape (batch=1, seq_len=N, input_size=1).
        
        Returns:
        y_pred: shape (batch=1, seq_len=N, 1).
        """
        # RNN output shape: (batch, seq_len, hidden_size)
        rnn_out, _ = self.rnn(x)  # ignoring hidden state
        # Map each hidden state to a single output
        y_pred = self.fc(rnn_out)  # shape (1, N, 1)
        return y_pred