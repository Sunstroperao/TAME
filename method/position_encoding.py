import math
import torch
import torch.nn as nn


class PositionalEncodingSine(nn.Module):

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 64
        self.temperature = temperature  # 10000

        self.scale = 2 * math.pi  # 2 * math.pi

    def forward(self, x):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # [0, 1, 2, ..., 63]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [10000^0, 10000^2/64, 10000^4/64, ...]
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        pos_x = x_embed[:, :, None] / dim_t  # [seq_len, batch_size, 1]
        pos_y = y_embed[:, :, None] / dim_t  # [seq_len, batch_size, 1]
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # [seq_len, batch_size, 64]
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)  # [seq_len, batch_size, 64]
        pos = torch.cat((pos_y, pos_x), dim=2)  # [seq_len, batch_size, 128]
        return pos


def build_position_encoding(args):
    if args.network == 'detr':
        position_embedding = PositionalEncodingSine(args.hidden_dim // 2, temperature=10000)
    else:
        position_embedding = PositionalEncodingSine(args.encoder_input_dim // 2, temperature=10000)

    return position_embedding


if __name__ == '__main__':
    a = torch.randn(12, 3)
    b = torch.randn(12, 3)
    print(torch.max(a, dim=1)[1])
    print(torch.argmax(a, dim=1) == torch.max(a, dim=1)[1])
    print(torch.sum(torch.argmax(a, dim=1) == torch.argmax(b, dim=1)).item())