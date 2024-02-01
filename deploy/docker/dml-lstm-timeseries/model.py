import torch.nn as nn


class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMTimeSeriesModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


import torch
import torch.nn as nn

class ResidualLSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(ResidualLSTMTimeSeriesModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size + input_size, output_size)

    def forward(self, input_seq):
        # LSTM output
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        lstm_out = lstm_out.view(len(input_seq), -1)

        combined_input = torch.cat((lstm_out, input_seq), dim=1)

        # Final prediction with residual connection
        predictions = self.linear(combined_input)
        return predictions
