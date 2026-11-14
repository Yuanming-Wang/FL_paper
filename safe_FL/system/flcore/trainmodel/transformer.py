# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, nlayers: int, num_classes: int, 
                 dropout: float = 0.1, max_len: int = 200, d_hid: int = 2048, pooling: str = 'mean'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pooling = pooling
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.hidden_dim = d_model
        self.fc = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, lengths: Tensor = None) -> Tensor:
        if isinstance(src, (tuple, list)):
            src, lengths = src
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            lengths: Tensor, valid sequence lengths for each batch item

        Returns:
            output Tensor of shape [batch_size, num_classes]
        """
        x = self.embedding(src) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)

        padding_mask = None
        if lengths is not None:
            lengths = lengths.to(src.device).clamp(min=1, max=src.size(1))
            valid = torch.arange(src.size(1), device=src.device).unsqueeze(0) < lengths.unsqueeze(1)
            padding_mask = ~valid
        else:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if self.pooling == 'last':
            x = x[torch.arange(x.size(0), device=x.device), valid.sum(dim=1) - 1]
        else:
            x = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        output = self.fc(x)

        return output
