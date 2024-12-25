import torch
import torch.nn as nn
from torch.nn import functional as F
from position_encoding import build_position_encoding
from transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, \
    Residual
from expert import Expert_Gate
from utils import out_activation
from einops import repeat


class HighwayNet(nn.Module):
    def __init__(self, args):
        super(HighwayNet, self).__init__()

        self.args = args

        # Initalize embeddings and input projection.
        self.input_embedding = nn.Linear(args.input_dim, args.encoder_input_dim)
        self.position_encoding = build_position_encoding(args)
        self.query_position_ecoding = nn.Embedding(args.num_queries, args.hidden_dim)

        # Initialize encoder and decoder transformer layers.
        encoder_layer = TransformerEncoderLayer(args.encoder_input_dim, args.nheads,
                                                dim_feedforward=args.dim_feedforward, dropout=0.1, activation=args.activation)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)

        decoder_layer = TransformerDecoderLayer(args.decoder_input_dim, args.nheads,
                                                dim_feedforward=args.dim_feedforward, dropout=0.1, activation=args.activation)
        decoder_norm = nn.LayerNorm(args.decoder_input_dim)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=args.dec_layers, 
                                            norm=decoder_norm, return_intermediate=args.return_intermediate)

        # Intialize Temporal attention mechanism.
        self.temporal_q = nn.Linear(args.input_dim, args.attn_nhead * args.attn_out)
        self.temporal_k = nn.Linear(args.input_dim, args.attn_nhead * args.attn_out)
        self.temporal_v = nn.Linear(args.input_dim, args.attn_nhead * args.attn_out)
        self.temporal_residual = Residual(args.encoder_input_dim)

        # Initialize social attention mechanism.
        self.qf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.kf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.vf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)

        # Initialize output projection.
        self.output_projection = nn.Linear(args.hidden_dim+6, args.pred_length * 5)
        self.output_lateral = nn.Linear(args.hidden_dim, args.lat_dim)
        self.output_longitudinal = nn.Linear(args.hidden_dim, args.lon_dim)

        # Initialize Mixture Experts module.
        self.expert_gate = Expert_Gate(args.hidden_dim) # 128

        # Initialize activation and regularization functions.
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask):

        if self.args.input_dim == 2:
            src = hist
        elif self.args.input_dim == 5:
            src = torch.cat((hist, cls, va), dim=-1)
            nbrs = torch.cat((nbrs, nbrscls, nbrsva), dim=-1)
        else:
            src = torch.cat((hist, cls, va, lane), dim=-1)
            nbrs = torch.cat((nbrs, nbrscls, nbrsva, nbrslane), dim=-1)

        # Temporal attention mechanism for temporal dependency.
        temporal_mask = temporal_mask.view(temporal_mask.size(0), temporal_mask.size(1) * temporal_mask.size(2), temporal_mask.size(3))
        temporal_mask = repeat(temporal_mask, 'b c n -> t b c n', t=self.args.hist_length)
        temporal_grid = torch.zeros_like(temporal_mask).float()
        temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs)

        temporal_query = self.temporal_q(src)
        temporal_query = torch.cat(torch.split(torch.unsqueeze(temporal_query, dim=2), int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 2, 3)
        temporal_key = self.temporal_k(temporal_grid)
        temporal_key = torch.cat(torch.split(temporal_key, int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 3, 2)
        temporal_value = self.temporal_v(temporal_grid)
        temporal_value = torch.cat(torch.split(temporal_value, int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 2, 3)
        temporal_attn_weights = torch.matmul(temporal_query, temporal_key)
        temporal_attn_weights /= torch.math.sqrt(self.args.attn_out)
        temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
        temporal_value = torch.matmul(temporal_attn_weights, temporal_value)
        temporal_value = torch.cat(torch.split(temporal_value, int(self.args.hist_length), dim=1), dim=-1).squeeze(2)
        temporal_value = self.temporal_residual(self.input_embedding(src).permute(1, 0, 2), temporal_value)

        hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src)), pos=self.position_encoding(src))
        hist_enc = hist_enc.permute(1, 0, 2)
        
        # Social attention mechanism for interaction capture.
        nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs)), pos=self.position_encoding(nbrs))
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b c n -> t b c n', t=self.args.hist_length)

        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)

        query = self.qf(hist_enc)
        _, _, embed_size = query.shape
        query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), int(embed_size / self.args.attn_nhead), dim=-1), dim=1)  # (batch_size, seq_len*attn_nhead, 1, att_out)
        key = torch.cat(torch.split(self.kf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(1, 0, 3, 2)  # (batch_size, seq_len*attn_nhead, att_out, 1)
        value = torch.cat(torch.split(self.vf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(1, 0, 2, 3)
        attn_weights = torch.matmul(query, key)
        attn_weights /= torch.math.sqrt(self.args.encoder_input_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        value = torch.matmul(attn_weights, value)
        value = torch.cat(torch.split(value, int(hist.shape[0]), dim=1), dim=-1).squeeze(2)

        # Temporal and social aggregation.
        temporal_spatial_agg = self.leaky_relu(temporal_value + value)

        enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1).permute(1, 0, 2)

        memory = enc
        query_pos = self.query_position_ecoding.weight.unsqueeze(1).repeat(1, enc.shape[1], 1)
        tgt = torch.zeros_like(query_pos)
        hs = self.decoder(tgt, memory, query_pos=query_pos)

        # Mixture of Experts module.
        expert_output = self.expert_gate(hs[-1])  # with mixture of experts module.
        # expert_output = hs[-1]  # without mixture of experts module.

        # Multi-task output projection.
        output_lat = F.softmax(self.output_lateral(expert_output), dim=-1)
        output_lon = F.softmax(self.output_longitudinal(expert_output), dim=-1)

        dec = torch.cat((expert_output, output_lat, output_lon), dim=-1)
        output = self.output_projection(dec).view(*hs.shape[1:3], self.args.pred_length, 5)
        output_traj = out_activation(output)

        return output_traj, output_lat, output_lon
