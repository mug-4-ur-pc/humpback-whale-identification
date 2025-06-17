import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class ArcMarginProduct(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: DictConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = cfg.scale
        self.margin = cfg.margin
        self.label_smoothing = cfg.label_smoothing
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = cfg.easy_margin
        self.cos_margin = math.cos(cfg.margin)
        self.sin_margin = math.sin(cfg.margin)
        self.th = math.cos(math.pi - cfg.margin)
        self.mm = math.sin(math.pi - cfg.margin) * cfg.margin

    def forward(self, embeddings, one_hot_label):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_margin - sine * self.sin_margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if one_hot_label is not None:
            if self.label_smoothing > 0:
                one_hot_label = (
                    1 - self.label_smoothing
                ) * one_hot_label + self.label_smoothing / self.out_features

            output = (one_hot_label * phi) + ((1.0 - one_hot_label) * cosine)
        else:
            output = cosine

        output *= self.scale

        return output
