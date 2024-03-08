import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, kernel_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()

        self.dynamic_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.static_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim*2,
                      out_channels=hidden_dim,
                      kernel_size=ks)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, 2)

    def forward(self, x):
        embedded = torch.cat((
             self.dynamic_embedding(x),
             self.static_embedding(x)), dim=2)
        embedded = embedded.permute(0, 2, 1)
        conved = [self.dropout(nn.functional.relu(conv(embedded))) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        outputs = self.fc(cat)
        return outputs
