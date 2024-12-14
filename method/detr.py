# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @File Name： model.py
# @Time     :  2023/5/30
# @Author   :  Jiang Hao
# @Mail     :  jianghaotbs@163.com
from position_encoding import build_position_encoding
from transformer import build_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import out_activation

class DetrModel(nn.Module):

    def __init__(self, args):
        super(DetrModel, self).__init__()
        self.args = args
        self.embed = nn.Linear(args.input_dim, args.hidden_dim)
        self.num_queries = args.num_queries
        self.hidden_dim = args.hidden_dim
        self.class_lat = nn.Linear(args.hidden_dim, args.lat_dim)
        self.class_lon = nn.Linear(args.hidden_dim, args.lon_dim)
        self.traj_embed = MLP(args.hidden_dim, args.hidden_dim, args.pred_length * 5, 3)
        self.query_embed = nn.Embedding(args.num_queries, args.hidden_dim)

        self.position_embedding = build_position_encoding(args)
        self.transformer = build_transformer(args)

    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):
        src = hist
        hs = self.transformer(self.embed(src), None, self.query_embed.weight, self.position_embedding(src))

        outputs_lat = F.softmax(self.class_lat(hs), dim=-1)
        outputs_lon = F.softmax(self.class_lon(hs), dim=-1)
        outputs_traj = self.traj_embed(hs).view(*hs.shape[:3], self.args.pred_length, 5)
        outputs_traj = out_activation(outputs_traj[-1])

        # out = {'pred_lat': outputs_lat[-1], 'pred_lon': outputs_lon[-1], 'pred_traj': outputs_traj}
        return outputs_traj, outputs_lat[-1], outputs_lon[-1]


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":
    a = torch.randn(2, 3, 4)
    print(a.transpose(0, 1).shape)
    print(a.transpose(1, 2) == a.permute(0, 2, 1))