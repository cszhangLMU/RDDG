import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Sequential, Sigmoid, Parameter


def Truncated_normal(a, b, mean=0, std=1):
    size = (a, b)
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2.5) & (tmp > -2.5)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super(ConditionalNorm, self).__init__()

        self.bn = BatchNorm1d(in_channel)
        self.fc = Linear(n_class, 20)

        self.embed = Linear(60, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, em_lable):
        a, b = em_lable.shape
        con1 = self.fc(em_lable)
        con2 = Truncated_normal(a, 40)
        con = torch.cat([con1, con2], 1)
        out = self.bn(input)
        embed = self.embed(con)
        gamma, beta = embed.chunk(2, 1)
        out = gamma * out + beta
        return out


class Residual(nn.Module):
    def __init__(self, i, o, n_class):
        super(Residual, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(o, int(o / 2))
        self.fc3 = Linear(int(o / 2), o)
        self.bn = ConditionalNorm(o, n_class)
        self.lerelu = LeakyReLU(0.2)

    def forward(self, input, em_lable):
        out1 = self.fc1(input)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = self.bn(out1 + out3, em_lable)
        out = self.lerelu(out)
        return out


class Block(nn.Module):
    def __init__(self, i, o):
        super(Block, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(o, int(o / 2))
        self.fc3 = Linear(int(o / 2), o)
        self.lerelu = LeakyReLU(0.2)
        self.dr = Dropout(0.5)

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = self.lerelu(out1 + out3)
        out = self.dr(out)
        return out


class Attention(nn.Module):
    def __init__(self, i, o):
        super(Attention, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(i, o // 4)
        self.fc3 = Linear(i, o // 4)
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        q = self.fc1(input)
        k = self.fc2(input)
        v = self.fc3(input)

        B, W = q.size()
        q = q.view(B, 1, 1 * W)  # query
        k = k.view(B, 1, 1 * W // 4)  # key
        v = v.view(B, 1, 1 * W // 4)  # value

        # attention weights
        w = F.softmax(torch.bmm(q.transpose(1, 2), k), -1)
        # attend and project
        o = torch.bmm(v, w.transpose(1, 2)).view(B, W)
        return self.gamma * o + input


class MLP(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(MLP, self).__init__()
        self.target_dim = target_dim
        dim = input_dim
        seq = []
        if input_dim >= 256:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 256)]
            seq += [Block(256, 128)]
            seq += [Block(128, 64)]
        elif input_dim >= 128:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 128)]
            seq += [Block(128, 64)]
        else:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 64)]

        dim = 64

        if self.target_dim >= 16:
            seq += [Linear(dim, 32)]
            dim = 32
        elif self.target_dim >= 8:
            seq += [Block(dim, 32)]
            seq += [Linear(32, 16)]
            dim = 16
        else:
            seq += [Block(dim, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]
            dim = 8

        seq += [Linear(dim, self.target_dim)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        output = self.seq(input)
        return output
