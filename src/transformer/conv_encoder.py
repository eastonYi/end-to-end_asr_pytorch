from collections import OrderedDict
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


class Conv1d(nn.Module):
    # the same as stack frames
    def __init__(self, d_input, d_hidden, n_layers, w_context, pad='same', name=''):
        super().__init__()
        assert n_layers >= 1
        self.n_layers = n_layers
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.w_context = w_context
        self.pad = pad
        self.name = name

        layers = [("{}/conv1d_0".format(name), nn.Conv1d(d_input, d_hidden, w_context, 1)),
                  # ("{}/batchnorm_0".format(name), nn.BatchNorm1d(d_hidden)),
                  ("{}/relu_0".format(name), nn.ReLU())]
        for i in range(n_layers-1):
            layers += [
                ("{}/conv1d_{}".format(name, i+1), nn.Conv1d(d_hidden, d_hidden, w_context, 1)),
                # ("{}/batchnorm_{}".format(name, i+1), nn.BatchNorm1d(d_hidden)),
                ("{}/relu_{}".format(name, i+1), nn.ReLU())
            ]
        layers = OrderedDict(layers)
        self.conv = nn.Sequential(layers)

    def forward(self, feats, feat_lengths):
        if self.pad == 'same':
            input_length = feats.size(1)
            feats = F.pad(feats, (0, 0, 0, self.n_layers * self.w_context))
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)

        if self.pad == 'same':
            tensor_length = input_length
            assert tensor_length <= outputs.size(1)
            outputs = outputs[:, :tensor_length, :]
            output_lengths = feat_lengths
        else:
            output_lengths = ((feat_lengths + sum(self.padding) -
                               1*(self.w_context-1)-1)/self.subsample + 1).long()

        return outputs, output_lengths


class Conv1dSubsample(nn.Module):
    # the same as stack frames
    def __init__(self, d_input, d_model, w_context, subsample, pad='same'):
        super().__init__()
        self.conv = nn.Conv1d(d_input, d_model, w_context, stride=subsample)
        self.conv_norm = LayerNorm(d_model)
        self.subsample = subsample
        self.w_context = w_context
        self.pad = pad

    def forward(self, feats, feat_lengths):
        if self.pad == 'same':
            input_length = feats.size(1)
            feats = F.pad(feats, (0, 0, 0, self.w_context))
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = outputs.permute(0, 2, 1)
        outputs = self.conv_norm(outputs)

        if self.pad == 'same':
            tensor_length = int(math.ceil(input_length/self.subsample))
            assert tensor_length <= outputs.size(1)
            outputs = outputs[:, :tensor_length, :]
            output_lengths = torch.ceil(feat_lengths*1.0/self.subsample).int()
        else:
            output_lengths = ((feat_lengths + sum(self.padding) - 1*(self.w_context-1)-1)/self.subsample + 1).long()

        return outputs, output_lengths


class Conv2dSubsample(nn.Module):
    def __init__(self, d_input, d_model, n_layers=2, pad='same'):
        super().__init__()
        assert n_layers >= 1
        self.n_layers = n_layers
        self.d_input = d_input
        self.pad = pad

        layers = [("subsample/conv0", nn.Conv2d(1, 32, 3, (2, 1))),
                  ("subsample/relu0", nn.ReLU())]
        for i in range(n_layers-1):
            layers += [
                ("subsample/conv{}".format(i+1), nn.Conv2d(32, 32, 3, (2, 1))),
                ("subsample/relu{}".format(i+1), nn.ReLU())
            ]
        layers = OrderedDict(layers)
        self.conv = nn.Sequential(layers)
        self.d_conv_out = int(math.ceil(d_input / 2))
        self.affine = nn.Linear(32 * self.d_conv_out, d_model)

    def forward(self, feats, feat_lengths):
        if self.pad == 'same':
            input_length = feats.size(1)
            feats = F.pad(feats, (0, 10, 0, 20))
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)[:, :, :, :self.d_conv_out]
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)

        if self.pad == 'same':
            output_lengths = feat_lengths
            tensor_length = input_length
            for _ in range(self.n_layers):
                output_lengths = torch.ceil(output_lengths / 2.0).int()
                tensor_length = int(math.ceil(tensor_length / 2.0))

            assert tensor_length <= outputs.size(1)
            outputs = outputs[:, :tensor_length, :]
        else:
            output_lengths = feat_lengths
            for _ in range(self.n_layers):
                output_lengths = ((output_lengths-1) / 2).long()

        outputs = self.affine(outputs)

        return outputs, output_lengths
