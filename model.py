import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

import common
from common import config

torch.manual_seed(0)


class Packer:
    def __init__(self):
        self._sorted_indices = None

    def pack_data(self, data: torch.Tensor, lens: torch.Tensor):
        sorted_lens, sorted_indices = torch.sort(lens, dim=0, descending=True)

        x = rnn_utils.pack_padded_sequence(data[sorted_indices], sorted_lens, batch_first=True)
        self._sorted_indices = sorted_indices
        return x

    def unpack_data(self, data: torch.Tensor):
        x, _ = rnn_utils.pad_packed_sequence(data, batch_first=True, total_length=common.LONGEST_SAMPLE)

        unsorted_indices = torch.sort(self._sorted_indices, dim=0)[1]
        x = x[unsorted_indices]
        x = x.contiguous()
        return x


class BaseTaggingModel(nn.Module):
    def __init__(self, tagset_size: int):
        super().__init__()
        self.tagset_size = tagset_size
        self.sorter = Packer()

    def get_parameter_number(self):
        return np.sum([np.prod(parameter.size()) for parameter in self.parameters()])

    def loss(self, y_pred, y_true, lens, masks):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1, self.tagset_size)

        nb_tokens = torch.sum(lens)
        mask = (y_true < common.PAD_TOKEN).float()
        y_true = torch.clamp(y_true, 0, config['tagset_size'] - 1)
        y_pred = y_pred[torch.arange(0, y_pred.size(0), device=common.DEVICE).long(), y_true.long()] * mask

        ce_loss = -torch.sum(y_pred) / nb_tokens
        return ce_loss

    def acc(self, y_pred, y_true, lens):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1, self.tagset_size)

        nb_tokens = torch.sum(lens)
        mask = (y_true < common.PAD_TOKEN).float()
        y_true = torch.clamp(y_true, 0, config['tagset_size'] - 1)
        y_pred = y_pred.argmax(dim=1)
        acc = ((y_true.long() == y_pred).float() * mask).sum()

        return acc / nb_tokens

    def fscore(self, y_pred, y_true, lens):
        fscores = torch.zeros((self.tagset_size,))
        weights = torch.zeros((self.tagset_size,))
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1, self.tagset_size)
        y_pred = y_pred.argmax(dim=1)
        for i in range(self.tagset_size):
            mask = ((y_true == i) & (y_true < common.PAD_TOKEN)).float()
            true_positives = (((y_true == i) & (y_pred == i)).float() * mask).sum()
            false_positives = (((y_true != i) & (y_pred == i)).float() * mask).sum()
            false_negatives = (((y_true == i) & (y_pred != i)).float() * mask).sum()
            class_num = mask.sum()
            precision = true_positives / (true_positives + false_positives + common.EPSILON)
            recall = true_positives / (true_positives + false_negatives + common.EPSILON)
            fscores[i] = 2 * precision * recall / (precision + recall + common.EPSILON)
            weights[i] = class_num
        return ((fscores * weights) / (weights.sum() + common.EPSILON)).sum()

    def predict(self, dataset):
        predictions = []
        for x, y, lens, masks in dataset:
            if common.USE_CUDA:
                x, y, lens, masks = x.cuda(), y.cuda(), lens.cuda(), masks.cuda()
            preds = self.forward(x, lens, masks).cpu().detach().numpy()
            for sample, length in zip(preds, lens):
                predictions.extend(sample[:length].reshape((-1, self.tagset_size)).tolist())
        return predictions


class PosTagger(BaseTaggingModel):
    def __init__(self,
                 tagset_size: int,
                 hidden_dim_1: int = config['hidden_dim_1'],
                 hidden_dim_2: int = config['hidden_dim_2'],
                 num_layers_1: int = config['num_layers_1'],
                 num_layers_2: int = config['num_layers_2']):
        super().__init__(tagset_size)
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.num_layers_1 = num_layers_1
        self.num_layers_2 = num_layers_2

        self.layer_1 = nn.GRU(
            input_size=config['input_size'],
            hidden_size=self.hidden_dim_1,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )
        self.layer_2 = nn.GRU(
            input_size=self.hidden_dim_1 * 2,
            hidden_size=self.hidden_dim_2,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim_2 * 2, self.tagset_size)
        )

        self.hidden_1 = None
        self.hidden_2 = None

    def init_hidden_1(self, batch_size):
        # h_0 (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.randn(self.num_layers_1 * 2, batch_size, self.hidden_dim_1, device=common.DEVICE))

    def init_hidden_2(self, batch_size):
        return Variable(torch.randn(self.num_layers_2 * 2, batch_size, self.hidden_dim_2, device=common.DEVICE))

    def forward(self, x, lens, *args):
        batch_size, seq_len, _ = x.size()

        self.hidden_1 = self.init_hidden_1(batch_size)
        self.hidden_2 = self.init_hidden_2(batch_size)

        x = self.sorter.pack_data(x, lens)
        output, _ = self.layer_1(x, self.hidden_1)
        output, _ = self.layer_2(output, self.hidden_2)
        x = self.sorter.unpack_data(output)

        seq_len = x.size(1)
        x = x.view(-1, x.size(2))
        x = self.classifier(x)

        x = torch.log_softmax(x, dim=1)
        x = x.view(batch_size, seq_len, self.tagset_size)
        return x


class PosTagger(BaseTaggingModel):
    def __init__(self,
                 tagset_size: int,
                 hidden_dim_1: int = config['hidden_dim_1'],
                 hidden_dim_2: int = config['hidden_dim_2'],
                 num_layers_1: int = config['num_layers_1'],
                 num_layers_2: int = config['num_layers_2']):
        super().__init__(tagset_size)
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.num_layers_1 = num_layers_1
        self.num_layers_2 = num_layers_2

        self.layer_1 = nn.GRU(
            input_size=config['input_size'],
            hidden_size=self.hidden_dim_1,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )
        self.layer_2 = nn.GRU(
            input_size=self.hidden_dim_1 * 2,
            hidden_size=self.hidden_dim_2,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )

        self.classifier = nn.Linear(self.hidden_dim_2 * 2, tagset_size)
        self.context_processing = nn.Linear(self.hidden_dim_2 * 2, self.hidden_dim_2 * 2)
        self.context_vector = nn.Parameter(torch.randn(self.hidden_dim_2 * 2, 1))

        self.hidden_1 = None
        self.hidden_2 = None

    def init_hidden_1(self, batch_size):
        # h_0 (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.randn(self.num_layers_1 * 2, batch_size, self.hidden_dim_1, device=common.DEVICE))

    def init_hidden_2(self, batch_size):
        return Variable(torch.randn(self.num_layers_2 * 2, batch_size, self.hidden_dim_2, device=common.DEVICE))

    def forward(self, x, lens, masks):
        batch_size, seq_len, _ = x.size()

        self.hidden_1 = self.init_hidden_1(batch_size)
        self.hidden_2 = self.init_hidden_2(batch_size)

        x = self.sorter.pack_data(x, lens)
        output, _ = self.layer_1(x, self.hidden_1)
        output, _ = self.layer_2(output, self.hidden_2)
        x = self.sorter.unpack_data(output)

        seq_len = x.size(1)
        # x = x.view(-1, x.size(2))
        u_vector = x.view(-1, x.size(2))
        u_vector = torch.tanh(self.context_processing(u_vector))
        pre_alphas = u_vector.mm(self.context_vector).view(*x.size()[:-1])
        pre_alphas = torch.exp(pre_alphas) * masks
        pre_alphas = pre_alphas / pre_alphas.sum(dim=1, keepdim=True)

        # processed_data =

        x = self.classifier(x)

        x = torch.log_softmax(x, dim=1)
        x = x.view(batch_size, seq_len, self.tagset_size)
        return x


class AttentionModel(BaseTaggingModel):
    def __init__(self,
                 tagset_size: int,
                 hidden_dim_1: int = config['hidden_dim_1'],
                 hidden_dim_2: int = config['hidden_dim_2'],
                 num_layers_1: int = config['num_layers_1'],
                 num_layers_2: int = config['num_layers_2'],
                 ):
        super().__init__(tagset_size)
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.num_layers_1 = num_layers_1
        self.num_layers_2 = num_layers_2

        self.layer_1 = nn.GRU(
            input_size=config['input_size'],
            hidden_size=self.hidden_dim_1,
            batch_first=True,
            bidirectional=True,
            num_layers=2
        )
        self.layer_2 = nn.GRU(
            input_size=self.hidden_dim_1 * 2,
            hidden_size=self.hidden_dim_2,
            batch_first=True,
            bidirectional=True,
            num_layers=1
        )

        self.classifier = nn.Linear(self.hidden_dim_2 * 2, tagset_size)
        self.context_processing = nn.Linear(self.hidden_dim_2 * 2, self.hidden_dim_2 * 2)
        self.context_vector = nn.Parameter(torch.randn(self.hidden_dim_2 * 2, 1))

        self.hidden_1 = None
        self.hidden_2 = None

    def init_hidden_1(self, batch_size):
        # h_0 (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.randn(self.num_layers_1 * 2, batch_size, self.hidden_dim_1, device=common.DEVICE))

    def init_hidden_2(self, batch_size):
        return Variable(torch.randn(self.num_layers_2 * 2, batch_size, self.hidden_dim_2, device=common.DEVICE))

    def forward(self, x, lens, masks):
        batch_size, seq_len, _ = x.size()

        self.hidden_1 = self.init_hidden_1(batch_size)
        self.hidden_2 = self.init_hidden_2(batch_size)

        x = self.sorter.pack_data(x, lens)
        output, _ = self.layer_1(x, self.hidden_1)
        output, _ = self.layer_2(output, self.hidden_2)
        x = self.sorter.unpack_data(output)

        seq_len = x.size(1)
        # x = x.view(-1, x.size(2))
        u_vector = x.view(-1, x.size(2))
        u_vector = torch.tanh(self.context_processing(u_vector))
        pre_alphas = u_vector.mm(self.context_vector).view(*x.size()[:-1])
        pre_alphas = torch.exp(pre_alphas) * masks
        pre_alphas = pre_alphas / pre_alphas.sum(dim=1, keepdim=True)

        # processed_data =

        x = self.classifier(x)

        x = torch.log_softmax(x, dim=1)
        x = x.view(batch_size, seq_len, self.tagset_size)
        return x


class SeModule(nn.Module):
    def __init__(self, input_filters: int, r: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(input_filters, input_filters // r),
            nn.ReLU(),
            nn.Linear(input_filters // r, input_filters),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, l = inputs.size()
        x = inputs.mean(-1)
        x = self.se(x)
        x = x.view(b, c, 1)
        return x * inputs


class BnConvo(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(input_filters, output_filters, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(output_filters),
        )

    def forward(self, x):
        return self.layers(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, num_dilations: int):
        super().__init__()
        self.num_dilations = num_dilations
        self.dilation_weighting = nn.Parameter(torch.randn(1, 1, 1, num_dilations + 1))
        self.key_strength = nn.Parameter(torch.randn(1,))
        self.gamma = nn.Parameter(torch.randn(1, ))
        self.dilation_layers = nn.ModuleList([
                                                 BnConvo(input_filters, output_filters, 3, dilation) for dilation in
                                                 range(1, num_dilations + 1)
                                             ] + [BnConvo(input_filters, output_filters, 1, 1)])
        self.out_relu = nn.Sequential(
            nn.ReLU(),
            SeModule(output_filters, r=16),
            nn.Dropout(0.6)
        )

    def forward(self, x):
        x = torch.stack([layer(x) for layer in self.dilation_layers], dim=-1)
        beta = self.softplus(self.key_strength)
        w = torch.softmax(self.dilation_weighting * beta, dim=-1)
        gamma = self.modified_softplus(self.gamma)
        w = w ** gamma
        w = w / w.sum(dim=-1, keepdim=True)
        return self.out_relu((x * w).sum(dim=-1))

    def modified_softplus(self, gamma):
        return self.softplus(gamma) + 1

    def softplus(self, beta):
        return torch.log(torch.exp(beta) + 1)

    def get_parameter_figure(self):
        fig = plt.figure()
        plt.plot(list(range(self.num_dilations + 1)), torch.softmax(self.dilation_weighting[0, 0, 0], 0).detach().cpu().numpy())
        return fig


class ResidualBlock(nn.Module):
    def __init__(self, input_filters: int, output_filters: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.convo = nn.Sequential(
            nn.Conv1d(input_filters, output_filters, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(output_filters),
            nn.ReLU(),
            nn.Conv1d(output_filters, output_filters, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(output_filters)
        )
        self.out_relu = nn.ReLU()
        self.se = SeModule(output_filters)

        if output_filters != input_filters:
            self.projection = nn.Sequential(
                nn.Conv1d(input_filters, output_filters, 1),
                nn.BatchNorm1d(output_filters)
            )
        else:
            self.projection = lambda x: x

    def forward(self, inputs):
        return self.out_relu(self.projection(inputs) + self.se(self.convo(inputs)))


class ConvoModel(BaseTaggingModel):
    def __init__(self,
                 tagset_size: int,
                 hidden_dim_1: int = config['hidden_dim_1'],
                 hidden_dim_2: int = config['hidden_dim_2'],
                 num_layers_1: int = config['num_layers_1'],
                 num_layers_2: int = config['num_layers_2'],
                 ):
        super().__init__(tagset_size)

        self.layers = [
            MultiScaleBlock(300, 128, 3),
            MultiScaleBlock(128, 128, 4),
            MultiScaleBlock(128, 128, 4),
            MultiScaleBlock(128, 128, 4),
            nn.Conv1d(128, self.tagset_size, 1)
        ]
        self.output_convo = nn.Sequential(*self.layers)

        # self.output_convo = nn.Sequential(
        #     nn.Conv1d(300, 128, 3, padding=1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     SeModule(128, r=16),
        #     nn.Conv1d(128, 128, 3, padding=2, dilation=2),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     SeModule(128, r=16),
        #     nn.Conv1d(128, 128, 3, padding=3, dilation=3),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     SeModule(128, r=16),
        #     nn.Conv1d(128, 128, 3, padding=4, dilation=4),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     SeModule(128, r=16),
        #     nn.Conv1d(128, self.tagset_size, 1)
        # )

    def forward(self, x, lens, masks):
        batch_size, seq_len, _ = x.size()

        # self.hidden_1 = self.init_hidden_1(batch_size)
        # self.hidden_2 = self.init_hidden_2(batch_size)
        #
        # x = self.sorter.pack_data(x, lens)
        # output, _ = self.layer_1(x, self.hidden_1)
        # output, _ = self.layer_2(output, self.hidden_2)
        # x = self.sorter.unpack_data(output)

        seq_len = x.size(1)
        x = x.permute(0, 2, 1)
        x = self.output_convo(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(-1, x.size(2))

        # x = self.classifier(x)

        x = torch.log_softmax(x, dim=1)
        x = x.view(batch_size, seq_len, self.tagset_size)
        return x
