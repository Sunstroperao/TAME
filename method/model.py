# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @File Name： detr_net.py
# @Time     :  2023/6/2
# @Author   :  Jiang Hao
# @Mail     :  jianghaotbs@163.com
import torch
import torch.nn as nn
from torch.nn import functional as F
from position_encoding import build_position_encoding
from transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, Residual
from utils import out_activation
import numpy as np
import math
from einops import rearrange, repeat


class HighwayNet(nn.Module):

    # Initialization Highway network
    def __init__(self, args):
        super(HighwayNet, self).__init__()
        self.args = args
        self.soc_embedding_dim = (
            ((args.grid_size[0] - 4) + 1) // 2) * args.conv_3x1_depth

        # position encoding
        self.pos = build_position_encoding(args)
        # learnable position encoding
        self.query_embed = nn.Embedding(args.num_queries, args.hidden_dim)
        # Input embedding
        self.ip_embed = nn.Linear(
            args.input_dim, args.encoder_input_dim)  # (2, 32)

        # Encoder Transformer
        self.encoder_layer = TransformerEncoderLayer(args.encoder_input_dim, args.nheads,
                                                     dim_feedforward=2048, dropout=0.1, activation=args.activation)

        self.encoder = TransformerEncoder(
            self.encoder_layer, num_layers=args.enc_layers)

        # Decoder Transformer
        self.decoder_layer = TransformerDecoderLayer(args.decoder_input_dim, args.nheads,
                                                     dim_feedforward=2048, dropout=0.1, activation=args.activation)
        self.decoder_norm = nn.LayerNorm(args.decoder_input_dim)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=args.dec_layers,
                                          norm=self.decoder_norm, return_intermediate=args.return_intermediate)

        # Social Attention Mechanism
        self.qf = nn.Linear(args.encoder_input_dim,
                            args.attn_nhead * args.attn_out)
        self.kf = nn.Linear(args.encoder_input_dim,
                            args.attn_nhead * args.attn_out)
        self.vf = nn.Linear(args.encoder_input_dim,
                            args.attn_nhead * args.attn_out)

        self.residual = Residual(args.encoder_input_dim)

        # Output layers:
        self.op = nn.Linear(args.hidden_dim + 6, args.pred_length * 5)
        # Lateral maneuver classification
        self.op_lat = nn.Linear(args.hidden_dim, args.lat_dim)
        # Longitudinal maneuver classification
        self.op_lon = nn.Linear(args.hidden_dim, args.lon_dim)

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)

    # Forward pass
    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):
        # Encode the history trajectory of the Agent vehicle
        if self.args.input_dim == 2:
            src = hist  # (seq_len, batch_size 2)
        elif self.args.input_dim == 5:
            src = torch.cat((hist, cls, va), dim=-1)  # (seq_len, batch_size 5)
            # (seq_len, batch_size 5)
            nbrs = torch.cat((nbrs, nbrscls, nbrsva), dim=-1)
        else:
            # (seq_len, batch_size 6)
            src = torch.cat((hist, cls, va, lane), dim=-1)
            nbrs = torch.cat((nbrs, nbrscls, nbrsva, nbrslane),
                             dim=-1)  # (seq_len, batch_size 6)

        hist_enc = self.encoder(self.leaky_relu(self.ip_embed(src)),
                                pos=self.pos(src))  # (seq_len, batch_size, encoder_input_dim)
        # (batch_size, seq_len, encoder_input_dim)
        hist_enc = hist_enc.permute(1, 0, 2)

        # Encode the history trajectories of the neighbor vehicles
        nbrs_enc = self.encoder(self.leaky_relu(self.ip_embed(nbrs)),
                                pos=self.pos(nbrs))  # (seq_len, batch_size, encoder_input_dim)
        mask = mask.view(mask.size(0), mask.size(1)
                         * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b c n -> t b c n', t=self.args.hist_length)

        # Scatter grid features
        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)

        # Social Attention Mechanism
        # (batch_size, seq_len, attn_nhead * att_out)
        query = self.qf(hist_enc)
        _, _, embed_size = query.shape
        query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), int(embed_size / self.args.attn_nhead), dim=-1),
                          dim=1)  # (batch_size, seq_len*attn_nhead, 1, att_out)
        key = torch.cat(torch.split(self.kf(soc_enc), int(embed_size / self.args.attn_nhead),
                                    dim=-1), dim=0).permute(1, 0, 3, 2)  # (batch_size, seq_len*attn_nhead, att_out, 1)
        value = torch.cat(torch.split(self.vf(soc_enc), int(embed_size / self.args.attn_nhead),
                                      dim=-1), dim=0).permute(1, 0, 2, 3)
        attn_weights = torch.matmul(query, key)
        attn_weights /= torch.math.sqrt(self.args.encoder_input_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        value = torch.matmul(attn_weights, value)
        value = torch.cat(torch.split(value, int(
            hist.shape[0]), dim=1), dim=-1).squeeze(2)

        # Gate

        # Residual connection
        # (batch_size, seq_len, encoder_input_dim)
        value = self.residual(hist_enc, value)

        # Concatenate ecoded history trajectory and social embedding
        # (seq_len, batch_size, encoder_input_dim)
        enc = torch.cat((value, hist_enc), dim=-1).permute(1, 0, 2)

        # Decode the future trajectory features
        memory = enc
        # learnable positional encoding: (num_queries, batch_size, hidden_dim)
        query_pos = self.query_embed.weight.unsqueeze(
            1).repeat(1, enc.shape[1], 1)
        tgt = torch.zeros_like(query_pos)
        hs = self.decoder(tgt, memory,
                          query_pos=query_pos)  # (num_decoder_layers, num_queries, batch_size, decoder_input_dim)

        # Output layers
        # (num_decoder_layers, num_queries, batch_size,
        output_lat = F.softmax(self.op_lat(hs), dim=-1)
        # num_lat_classes)
        # (num_decoder_layers, num_queries, batch_size,
        output_lon = F.softmax(self.op_lon(hs), dim=-1)
        # num_lon_classes)

        # trajectory prediction
        dec = torch.cat((hs, output_lat, output_lon), dim=-1)
        output_traj = self.op(dec).view(*hs.shape[:3], self.args.pred_length,
                                        5)  # (num_decoder_layers, num_queries, batch_size, pred_length, 5)
        # (num_queries, batch_size, pred_length, 2)
        output_traj = out_activation(output_traj[-1])

        return output_traj, output_lat[-1], output_lon[-1]
