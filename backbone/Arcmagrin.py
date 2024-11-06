import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, num, s=20.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.label = torch.eye(num).unsqueeze(0)
        # self.label = torch.arange(num, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    def forward(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input [B, N, L]
        Bs = input.size(0)
        label = self.label.repeat(Bs, 1, 1).to(input.device)
        cosine = F.linear(F.normalize(input, dim=-1), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # cos(\theta) > 0 phi取cos(\theta - \m)
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (label * phi) + ((1.0 - label) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        output = torch.exp(output)
        loss = torch.sum(label * output, dim=-1) / torch.sum(output, dim=-1)
        loss = -torch.log(loss)
        # print(output)

        return torch.mean(loss)


class ArcMarginProduct_PartImage(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, num, s=20.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct_PartImage, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.num_part = num
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.label = torch.eye(num).unsqueeze(0)
        # self.label = torch.arange(num, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    def forward(self, input, label_index):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input [B, N, L]
        Bs = input.size(0)

        label_index = label_index.unsqueeze(1).repeat(1, self.num_part) * self.num_part + torch.arange(self.num_part, device=label_index.device).unsqueeze(0).long()
        label_index = label_index.flatten(0)

        weight = torch.index_select(self.weight, 0, label_index)
        weight = weight.view(Bs, self.num_part, -1)

        # label =F.one_hot(label, num_classes=self.out_features)
        label = self.label.repeat(Bs, 1, 1).to(input.device)
        # cosine -> [B, N, N]
        # cosine = F.linear(F.normalize(input, dim=-1), F.normalize(self.weight))

        cosine = torch.einsum("bqc,bcl->bql", F.normalize(input, dim=-1), F.normalize(weight, dim=-1).permute(0, 2, 1))

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # cos(\theta) > 0 phi取cos(\theta - \m)
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (label * phi) + ((1.0 - label) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        output = torch.exp(output)
        loss = torch.sum(label * output, dim=-1) / torch.sum(output, dim=-1)
        loss = -torch.log(loss)
        # print(output)

        return torch.mean(loss)
