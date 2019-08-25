# import torch
import torch.nn as nn


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 lstm_size=64,
                 lstm_num_layers=1,
                 drop_out_rate=0.1,
                 num_of_classes=9292):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_out_rate = drop_out_rate
        self.num_of_classes = num_of_classes
        self.use_batch = False
        self.create_params()
        return

    def create_params(self):
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.lstm_size,
                            num_layers=self.lstm_num_layers)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.fc1 = nn.Linear(self.lstm_size, self.num_of_classes, bias=False)
        self.reset_params()
        return

    def batch(self, use_batch):
        self.use_batch = use_batch
        if self.use_batch:
            self.lstm.batch_first = True
        else:
            self.lstm.batch_first = False
        return

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
        nn.init.uniform_(self.lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self, input_tensor):
        if self.use_batch:
            lstm_out, (last_hidden_state, last_cell_state) = self.lstm(input_tensor[0])
        else:
            lstm_out, (last_hidden_state, last_cell_state) = self.lstm(input_tensor[0].view(input_tensor[0].size(1), 1, -1))
        out = self.dropout(last_hidden_state)
        out = self.fc1(out)
        out = out.view(-1, self.num_of_classes)
        return out