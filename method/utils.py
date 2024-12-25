import torch
from matplotlib import pyplot as plt

# Loss function for training
def detr_loss(pred, gt, mask):
    acc = torch.zeros_like(mask)  # [seq_len, batch_size 2]
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_quries, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]

    muX = pred[:, :, 0]  # [batch_size, seq_len]
    muY = pred[:, :, 1]  # [batch_size, seq_len]
    sigX = pred[:, :, 2]  # [batch_size, seq_len]
    sigY = pred[:, :, 3]  # [batch_size, seq_len]
    rho = pred[:, :, 4]  # [batch_size, seq_len]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)  # [batch_size, seq_len]
    x = gt[:, :, 0]  # [batch_size, seq_len]
    y = gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * (
            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
            - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - \
          torch.log(sigX * sigY * ohr) - 1.8379  # [batch_size, seq_len]
    acc[:, :, 0] = out.permute(1, 0)  # [seq_len, batch_size]
    acc[:, :, 1] = out.permute(1, 0)  # [seq_len, batch_size]
    acc = acc * mask  # [seq_len, batch_size, 2]
    loss_mse = torch.sum(acc) / torch.sum(mask)

    return loss_mse, nearest_mode_ids, nearest_mode_bs_ids


# Outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def detr_test_mse(pred, gt, mask):
    acc = torch.zeros_like(mask)
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_modes, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]
    muX = pred[:, :, 0]
    muY = pred[:, :, 1]
    x = gt[:, :, 0]
    y = gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out.permute(1, 0)
    acc[:, :, 1] = out.permute(1, 0)
    acc = acc * mask
    loss_mse = torch.sum(acc) / torch.sum(mask)
    return loss_mse, nearest_mode_ids, nearest_mode_bs_ids


def detr_val_mse(pred, gt, mask):
    acc = torch.zeros_like(mask)
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_modes, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]
    muX = pred[:, :, 0]
    muY = pred[:, :, 1]
    x = gt[:, :, 0]
    y = gt[:, :, 1]
    out_rmse = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)  # [batch_size, seq_len]
    out_de = torch.sqrt(out_rmse)  # [batch_size, seq_len]
    acc[:, :, 0] = out_rmse.permute(1, 0)
    acc[:, :, 1] = out_de.permute(1, 0)
    acc = acc * mask
    loss_val_rmse = torch.sum(acc[:, :, 0], dim=1)
    loss_val_de = torch.sum(acc[:, :, 1], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return loss_val_rmse, loss_val_de, counts, nearest_mode_ids, nearest_mode_bs_ids


# Custom activation for output layer (Graves, 2015)
def out_activation(x):
    muX = x[:, :, :, 0:1]
    muY = x[:, :, :, 1:2]
    sigX = x[:, :, :, 2:3]
    sigY = x[:, :, :, 3:4]
    rho = x[:, :, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=3)
    return out

