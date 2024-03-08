from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=300, num_hiddens=300, num_layers=2, dropout=0.3):
        super(LSTM, self).__init__()
        self.dynamic_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens,
                            num_layers=num_layers,
                            bidirectional=True)
        self.fc = nn.Linear(num_hiddens*2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embedded = self.dynamic_embedding(inputs.permute(1, 0))
        outputs, (hidden, cell) = self.lstm(embedded)  # output, (h, c)
        hidden = (torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        outs = self.fc(hidden)
        return outs
