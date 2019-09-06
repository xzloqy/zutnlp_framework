#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:47
# @Author  : Xinxin Zhang
"""
just one layer LSTM example
        LSTM --> SEG
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

from base.model.base_model import BaseModel


class EncoderModel(BaseModel):
    """Model include a transducer to predict at each time steps"""
    def __init__(self,
                 ntoken,
                 emsize,
                 nhid,
                 dropout=0.2,
                 rnn_type='LSTM',
                 bi=True):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, emsize)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.bi = bi

        # Select RNN cell type from LSTM, GRU, and Elman
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emsize, nhid, num_layers=1, bidirectional=bi)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(emsize, nhid, num_layers=1, bidirectional=bi)
        else:
            self.rnn = nn.RNN(emsize, nhid, num_layers=1, bidirectional=bi)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)

    def lstm_batch(self, inputs, hidden, sent=True):
        input, lengths = inputs
        if sent:
            embedded = self.drop(self.embed(input))
        else:
            embedded = self.drop(input)

        batch_size = input.size(0)
        num_valid = lengths.gt(0).int().sum().item()
        sorted_lengths, indices = lengths.sort(descending=True)
        lstm_inputs = embedded.index_select(0, indices)

        lstm_inputs = nn.utils.rnn.pack_padded_sequence(  # B x T x *
            lstm_inputs[:num_valid],
            sorted_lengths[:num_valid].tolist(),
            batch_first=True)
        if hidden is not None:
            hidden = (hidden[0].index_select(
                1,
                indices)[:, :num_valid].contiguous(), hidden[1].index_select(
                    1, indices)[:, :num_valid].contiguous())

        self.rnn.flatten_parameters()
        outputs, last_hidden = self.rnn(lstm_inputs, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)

        if num_valid < batch_size:
            zeros = outputs.new_zeros(batch_size - num_valid, outputs.size(1),
                                      self.nhid * (1 + int(self.bi)))
            outputs = torch.cat([outputs, zeros], dim=0)

            # zeros = last_hidden[0].new_zeros(
            #     2, batch_size - num_valid, self.nhid)
            #
            # last_hidden = (torch.cat([last_hidden[0], zeros], dim=1),
            #                torch.cat([last_hidden[1], zeros], dim=1))

        _, inv_indices = indices.sort()
        outputs = outputs.index_select(0, inv_indices)
        outputs = self.drop(outputs)

        # last_hidden = (last_hidden[0].index_select(1, inv_indices),
        #                last_hidden[1].index_select(1, inv_indices))

        return outputs

    def forward(self, inputs, hidden):
        outputs = self.lstm_batch(inputs, hidden)
        return outputs

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(
            weight.new((1 + int(self.bi)), batch_size, self.nhid).zero_()),
                Variable(
                    weight.new((1 + int(self.bi)), batch_size,
                               self.nhid).zero_()))


class LinearDecoder(BaseModel):
    """Linear decoder to decoder the outputs from the RNN Encoder.
        Then we can get the results of different tasks."""
    def __init__(self, nhid, ntags, bi=False):
        super().__init__()
        self.linear = nn.Linear(nhid * (1 + int(bi)), ntags)
        self.init_weights()
        self.nin = nhid
        self.nout = ntags
        self.bi = bi

    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, input):
        logit = self.linear(
            input.view(input.size(0) * input.size(1), input.size(2)))
        return logit.view(input.size(0), input.size(1), logit.size(1))


class SingleModel(BaseModel):
    """Joint Model to joint training two tasks.
       You can also only select one train mode to train one task.
       For args to specified the detail of training, include the task
       output and which layer we put it in. Number of tag first and
       then number of layer."""
    def __init__(self,
                 ntoken,
                 emsize,
                 nhid,
                 n_tags,
                 dropout=0.2,
                 rnn_type='LSTM',
                 bi=True,
                 train_mode='Joint'):
        super().__init__()
        self.ntoken = ntoken
        self.emsize = emsize
        self.nhid = nhid
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.bi = bi
        self.train_mode = train_mode

        # According to train type, take arguments
        self.n_tags = n_tags

        self.rnn = EncoderModel(ntoken, emsize, nhid, dropout, rnn_type, bi)
        # Decoders for two tasks
        self.linear = LinearDecoder(nhid, self.n_tags, bi)

    def forward(self, input, *hidden):
        logits = self.rnn(input, hidden[0])
        outputs = self.linear(logits)
        return outputs
