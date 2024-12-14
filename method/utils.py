# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @File Name： detr_utils.py
# @Time     :  2023/5/31
# @Author   :  Jiang Hao
# @Mail     :  jianghaotbs@163.com

import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import random
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
import concurrent.futures

sns.set_style("white")


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


# seed everything
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


def plot_traj(fut_pred, fut, hist, idx, idx_b):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(0, 90, 10):
        pred_x = fut_pred[idx, idx_b][i, :, 0].cpu().detach().numpy()
        pred_y = fut_pred[idx, idx_b][i, :, 1].cpu().detach().numpy()
        gt_x = fut[:, i, 0].cpu().detach().numpy()
        gt_y = fut[:, i, 1].cpu().detach().numpy()
        hist_x = hist[:, i, 0].cpu().detach().numpy()
        hist_y = hist[:, i, 1].cpu().detach().numpy()
        mul_x1 = fut_pred[0][i, :, 0].cpu().detach().numpy()
        mul_y1 = fut_pred[0][i, :, 1].cpu().detach().numpy()
        mul_x2 = fut_pred[1][i, :, 0].cpu().detach().numpy()
        mul_y2 = fut_pred[1][i, :, 1].cpu().detach().numpy()
        mul_x3 = fut_pred[2][i, :, 0].cpu().detach().numpy()
        mul_y3 = fut_pred[2][i, :, 1].cpu().detach().numpy()
        mul_x4 = fut_pred[3][i, :, 0].cpu().detach().numpy()
        mul_y4 = fut_pred[3][i, :, 1].cpu().detach().numpy()
        mul_x5 = fut_pred[4][i, :, 0].cpu().detach().numpy()
        mul_y5 = fut_pred[4][i, :, 1].cpu().detach().numpy()
        mul_x6 = fut_pred[5][i, :, 0].cpu().detach().numpy()
        mul_y6 = fut_pred[5][i, :, 1].cpu().detach().numpy()
        mul_x7 = fut_pred[6][i, :, 0].cpu().detach().numpy()
        mul_y7 = fut_pred[6][i, :, 1].cpu().detach().numpy()
        mul_x8 = fut_pred[7][i, :, 0].cpu().detach().numpy()
        mul_y8 = fut_pred[7][i, :, 1].cpu().detach().numpy()
        mul_x9 = fut_pred[8][i, :, 0].cpu().detach().numpy()
        mul_y9 = fut_pred[8][i, :, 1].cpu().detach().numpy()

        i = i // 10
        ax[i // 3, i % 3].set_xlim(-20, 20)
        ax[i // 3, i % 3].scatter(pred_x, pred_y, c='r', label='Prediction')
        ax[i // 3, i % 3].scatter(gt_x, gt_y, c='b', label='Ground Truth')
        ax[i // 3, i % 3].scatter(hist_x, hist_y, c='g', label='History')
        ax[i // 3, i % 3].scatter(mul_x1, mul_y1, c='y', label='mul1')
        ax[i // 3, i % 3].scatter(mul_x2, mul_y2, c='c', label='mul2')
        ax[i // 3, i % 3].scatter(mul_x3, mul_y3, c='k', label='mul3')
        ax[i // 3, i % 3].scatter(mul_x4, mul_y4, c='m', label='mul4')
        ax[i // 3, i % 3].scatter(mul_x5, mul_y5, c='purple', label='mul5')
        ax[i // 3, i % 3].scatter(mul_x6, mul_y6, c='grey', label='mul6')
        ax[i // 3, i % 3].scatter(mul_x7, mul_y7, c='orange', label='mul7')
        ax[i // 3, i % 3].scatter(mul_x8, mul_y8, c='pink', label='mul8')
        ax[i // 3, i % 3].scatter(mul_x9, mul_y9, c='brown', label='mul9')
        ax[i // 3, i % 3].legend()

    return fig

# Get the number of neighbors and its trajectory for each batch
def get_nbrs_trajectory(nbrs, nbrs_num_batch, b_index):
    nbr_num = nbrs_num_batch[b_index]
    start_index = int(sum(nbrs_num_batch[:b_index]))
    end_index = start_index + int(nbr_num)
    return nbrs[:, start_index:end_index, :]

# ====================Single Thread Version====================
# def plot_traj_heatmap(fut_pred, fut, hist, idx, idx_b, nbrs, nbrs_num_batch, fig_num):
#     fig, axes = plt.subplots(fig_num, fig_num, figsize=(18, 6))
#     plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjust space between subplots
#     num_plots = fig_num * fig_num
#     for i in range(num_plots):
#         ax = axes[i // fig_num, i % fig_num]
#         traj_index = i * (128 // fig_num // fig_num)
        
#         pred = fut_pred[idx, idx_b][traj_index, :, :2].cpu().detach().numpy()
#         gt = fut[:, traj_index, :2].cpu().detach().numpy()
#         history = hist[:, traj_index, :2].cpu().detach().numpy()
#         # multiple_pred = [fut_pred[batch][traj_index, :, :2].cpu().detach().numpy() for batch in range(9)]  # multiple predictions
        
#         # Plot highway lanes
#         ax.set_ylim(-20, 20)
#         # ax.set_facecolor("lightgrey")
#         for y in [6, 18, -6, -18]:
#             ax.axhline(y=y, linestyle='--', c='black', zorder=2)

#         # plot ground truth, history, prediction
#         ax.plot(gt[:, 1], gt[:, 0], marker='o', markersize=4, linewidth=3, c='b', label='Ground Truth', zorder=2)
#         ax.plot(history[:, 1], history[:, 0], marker='o', markersize=4, linewidth=3, c='g', label='History', zorder=2)
#         ax.plot(pred[:, 1], pred[:, 0], marker='o', markersize=4, linewidth=3, c='r', label='Prediction', zorder=2)  
        
        # plot multi-modal predicted trajectory
        # colors = ['cyan', 'magenta', 'yellow', 'lime', 'saddlebrown', 'blue', 'orange', 'purple', 'crimson']
        # for mul_pred, color in zip(multiple_pred, colors):
        #     # ax.scatter(mul_pred[:, 1], mul_pred[:, 0], c=color, s=10, zorder=2)
        #     ax.plot(mul_pred[:, 1], mul_pred[:, 0], c='purple', linestyle='--', zorder=2)
        
#         # plot target vehicle
#         x_lim =  ax.get_xlim()
#         x_range = x_lim[1] - x_lim[0]
#         scale = x_range / 1400 # compute the scale
#         rect = patches.Rectangle((-40 * scale, -1.5), 80 * scale, 3, linewidth=2, edgecolor='#A12929', facecolor='#A12929', zorder=4)
#         ax.add_patch(rect)
        
#         # plot neighbor vehicles and their trajectory
#         nbrs_traj = get_nbrs_trajectory(nbrs, nbrs_num_batch, traj_index)
#         if nbrs_traj.shape[1] != 0:
#             nbrs_traj = nbrs_traj.cpu().detach().numpy()
#             for j in range(int(nbrs_num_batch[traj_index])):
#                 # ax.scatter(nbrs_traj[:, j, 1], nbrs_traj[:, j, 0], c='dimgray', s=30, zorder=2)
#                 ax.plot(nbrs_traj[:, j, 1], nbrs_traj[:, j, 0], marker='o', markersize=4, linewidth=3, c='dimgray', zorder=2) 
                
#                 # Plot neighbor vehicle
#                 last_nbr_point = nbrs_traj[-1, j, :2]
#                 rect_nbr = patches.Rectangle((last_nbr_point[1] - 40 * scale, last_nbr_point[0] - 1.5), 80 * scale, 3, linewidth=2, edgecolor='#19196B', facecolor='#19196B', zorder=4)
#                 ax.add_patch(rect_nbr)
        
#         # NOTE: The following code is used to generate heatmap around the predicted trajectory
#         # ==============Calculate tangents and normals for predicted trajectory================
#         # 计算切线方向
#         tangents = np.diff(pred, axis=0)
#         tangents = np.vstack((tangents, tangents[-1]))  # 最后一个点的切线方向和前一个相同
#         # 垂线方向
#         normals = np.zeros_like(tangents)
#         normals[:, 0] = -tangents[:, 1]
#         normals[:, 1] = tangents[:, 0]
#         # 归一化垂线方向
#         normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
#         # Generate points around the predicted trajectory in a triangle shape 
#         num_points_per_side = np.linspace(0, 100, pred.shape[0])  # 每侧生成的点数
#         max_distance = np.linspace(0, 1.6, pred.shape[0])  # 最大距离
        
#         additional_points = []
#         for l in range(len(pred)):
#             for j in range(int(num_points_per_side[l])):
#                 distance = max_distance[l] * (j / num_points_per_side[l])
#                 additional_points.append(pred[l] + distance * normals[l])
#                 additional_points.append(pred[l] - distance * normals[l])
#         additional_points = np.array(additional_points)
        
#         # Add noise to the generated points
#         noise_std = (np.exp(np.linspace(0, 1, additional_points.shape[0])) - 1)
#         x = additional_points[:, 0] + np.random.normal(0, noise_std, additional_points.shape[0])
#         y = additional_points[:, 1] + np.random.normal(0, noise_std, additional_points.shape[0])   
#         # ==============Calculate tangents and normals for predicted trajectory================
        
#         # Smooth using Gaussian KDE
#         data = np.vstack([x, y])
#         kde = gaussian_kde(data, bw_method=0.6)  # adjust the bandwidth
        
#         # Generate grid for heatmap
#         x_grid = np.linspace(-20, 20, 200)
#         y_grid = np.linspace(x_lim[0], x_lim[1], 200)
#         X, Y = np.meshgrid(x_grid, y_grid)
#         Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
#         # Plot heatmap
#         threshold = 0.05  # set threshold for contour to cut off the low density area
#         levels = np.linspace(Z.min() + threshold * (Z.max() - Z.min()), Z.max(), 100)
#         ax.contourf(Y, X, Z, levels=levels, cmap='OrRd', zorder=1)
        
#         # hide axis and labels
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel('')
#         ax.set_ylabel('')
        
#         # Hide the surrounding box of each subplot
#         # for spine in ax.spines.values():
#         #     spine.set_visible(False)
    
#     return fig
# ====================Single Thread Version====================

# ====================Multi-Thread Version====================

def plot_single_traj_heatmap(ax, fut_pred, fut, hist, idx, idx_b, nbrs, nbrs_num_batch, traj_index):
    pred = fut_pred[idx, idx_b][traj_index, :, :2].cpu().detach().numpy()
    gt = fut[:, traj_index, :2].cpu().detach().numpy()
    history = hist[:, traj_index, :2].cpu().detach().numpy()
    multiple_pred = [fut_pred[batch][traj_index, :, :2].cpu().detach().numpy() for batch in range(9)]  # multiple predictions
    

    # ax.set_ylim(-20, 20)
    # ax.set_xlim(-400, 650)
    # for y in [6, 18, -6, -18]:
    #     ax.axhline(y=y, linestyle='--', linewidth=1, c='black', zorder=2)
    
    # ax.plot(gt[:, 1], gt[:, 0], marker='o', markersize=2.5, linewidth=2, c='b', label='Ground Truth', zorder=3)
    # ax.plot(history[:, 1], history[:, 0], marker='o', markersize=2.5, linewidth=2, c='g', label='History', zorder=2)
    # ax.plot(pred[:, 1], pred[:, 0], marker='o', markersize=4, linewidth=3, c='r', label='Prediction', zorder=2)  
    
    
    # plot multi-modal predicted trajectory
    # colors = ['cyan', 'magenta', 'yellow', 'lime', 'saddlebrown', 'blue', 'orange', 'purple', 'crimson']
    # for mul_pred, color in zip(multiple_pred, colors):
    #     # ax.scatter(mul_pred[:, 1], mul_pred[:, 0], c=color, s=10, zorder=2)
    #     ax.plot(mul_pred[:, 1], mul_pred[:, 0], marker='o', markersize=2.5, linewidth=2, c='purple', zorder=2)
    
    # x_lim = ax.get_xlim()
    # x_range = x_lim[1] - x_lim[0]
    # scale = x_range / 1400
    # rect = patches.Rectangle((-40 * scale, -1.5), 80 * scale, 3, linewidth=2, edgecolor='#A12929', facecolor='#A12929', zorder=4)
    # ax.add_patch(rect)
    
    # nbrs_traj = get_nbrs_trajectory(nbrs, nbrs_num_batch, traj_index)
    # if nbrs_traj.shape[1] != 0:
    #     nbrs_traj = nbrs_traj.cpu().detach().numpy()
    #     for j in range(int(nbrs_num_batch[traj_index])):
    #         ax.plot(nbrs_traj[:, j, 1], nbrs_traj[:, j, 0], marker='o', markersize=2.5, linewidth=2, c='dimgray', zorder=2) 
            # last_nbr_point = nbrs_traj[-1, j, :2]
            # rect_nbr = patches.Rectangle((last_nbr_point[1] - 40 * scale, last_nbr_point[0] - 1.5), 80 * scale, 3, linewidth=2, edgecolor='#19196B', facecolor='#19196B', zorder=4)
            # ax.add_patch(rect_nbr)
    
    tangents = np.diff(pred, axis=0)
    tangents = np.vstack((tangents, tangents[-1]))
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    num_points_per_side = np.linspace(0, 100, pred.shape[0])
    max_distance = np.linspace(0, 1.6, pred.shape[0])
    
    additional_points = []
    for l in range(len(pred)):
        for j in range(int(num_points_per_side[l])):
            distance = max_distance[l] * (j / num_points_per_side[l])
            additional_points.append(pred[l] + distance * normals[l])
            additional_points.append(pred[l] - distance * normals[l])
    additional_points = np.array(additional_points)
    
    noise_std = (np.exp(np.linspace(0, 1, additional_points.shape[0])) - 1)
    x = additional_points[:, 0] + np.random.normal(0, noise_std, additional_points.shape[0])
    y = additional_points[:, 1] + np.random.normal(0, noise_std, additional_points.shape[0])
    
    data = np.vstack([x, y])
    kde = gaussian_kde(data, bw_method=0.6)
    
    

    x_grid = np.linspace(-20, 20, 200)
    y_grid = np.linspace(-400, 1000, 200)
    # y_grid = np.linspace(x_lim[0], x_lim[1], 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    threshold = 0.05
    levels = np.linspace(Z.min() + threshold * (Z.max() - Z.min()), Z.max(), 100)
    ax.contourf(Y, X, Z, levels=100, cmap='viridis', zorder=1)
    
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Hide the surrounding box of each subplot
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_traj_heatmap(fut_pred, fut, hist, idx, idx_b, nbrs, nbrs_num_batch, fig_num):
    num_plots = fig_num * fig_num
    traj_indices = [i * (128 // fig_num // fig_num) for i in range(num_plots)]
    
    fig, axes = plt.subplots(fig_num, fig_num, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(num_plots):
            ax = axes[i // fig_num, i % fig_num]
            traj_index = traj_indices[i]
            futures.append(executor.submit(plot_single_traj_heatmap, ax, fut_pred, fut, hist, idx, idx_b, nbrs, nbrs_num_batch, traj_index))
        
        concurrent.futures.wait(futures)
    
    return fig

# ====================Multi-Thread Version====================
