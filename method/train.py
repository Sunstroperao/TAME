from tqdm import tqdm
from config import get_args_parser
from ngsim_dataset import NgsimDataset
from utils import detr_loss, detr_test_mse
from torch.utils.data import DataLoader
import time

import torch.nn as nn
from highwaynet import HighwayNet
import torch
import os
import sys
sys.path.append('..')

args = get_args_parser().parse_args()  # get the args of HighwayNet

# Set the args
args.network = 'highwaynet'
args.dataset = 'ngsim'
args.input_dim = 6


# Set the GPU
CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)


# Initialize the network
if args.network == 'highwaynet':
    net = HighwayNet(args)  # initialize the HighwayNet


# print the number of parameters
print('参数数目：', sum(param.numel() for param in net.parameters()))
if args.use_cuda:
    net = net.cuda()


# Initialize the optimizer and loss
optimizer = torch.optim.Adam(net.parameters())
cross_entropy = nn.BCELoss()

# Initialize the dataset
root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get the root path
if args.dataset == 'ngsim':
    trSet_path = os.path.join(root_path, 'data/ngsimdata/TrainSet.mat')
    valSet_path = os.path.join(root_path, 'data/ngsimdata/ValSet.mat')
elif args.dataset == 'highd':
    trSet_path = os.path.join(root_path, 'data/highddata/TrainSet.mat')
    valSet_path = os.path.join(root_path, 'data/highddata/ValSet.mat')
trSet = NgsimDataset(trSet_path)
valSet = NgsimDataset(valSet_path)

trDataloader = DataLoader(trSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=valSet.collate_fn)

# Start training and validation
step = 0
for epoch in range(args.epochs):
    print("**********************" + 'Training Epoch:', str(epoch) + "**********************")
    avg_train_loss = 0
    avg_train_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    net.train()
    for i, data in enumerate(tqdm(trDataloader)):
        start_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, _, _, cls, nbrscls, _, _, temporal_mask = data

        # Move the data to GPU
        if args.use_cuda:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
            va = va.cuda()
            nbrsva = nbrsva.cuda()
            lane = lane.cuda()
            nbrslane = nbrslane.cuda()
            cls = cls.cuda()
            nbrscls = nbrscls.cuda()
            temporal_mask = temporal_mask.cuda()

        optimizer.zero_grad()
        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask)
        loss_mse, index, b_index = detr_loss(fut_pred, fut, op_mask)
        loss = loss_mse + cross_entropy(lat_pred[index, b_index], lat_enc) + cross_entropy(lon_pred[index, b_index], lon_enc)

        loss.backward()
        # gradient clipping only used in HighwayNet
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        batch_time = time.time() - start_time  # each batch_time
        avg_train_time += batch_time  # total batch_time
        avg_train_loss += loss.item()  # total loss

        # Calculate the accuracy
        avg_lat_acc += torch.sum(
            torch.argmax(lat_pred[index, b_index], dim=1) == torch.argmax(lat_enc, dim=1)).item() / args.batch_size
        avg_lon_acc += torch.sum(
            torch.argmax(lon_pred[index, b_index], dim=1) == torch.argmax(lon_enc, dim=1)).item() / args.batch_size

        # print the loss and accuracy every 100 iterations
        if (i + 1) % 100 == 0:
            print(
                'Epoch: {}, Iteration: {:.2f}, Avg train loss: {:.3f}, Lat_Acc: {:.3f}, Lon_Acc: {:.3f}, Time: {:.3f}'
                .format(epoch, i / (len(trSet) / args.batch_size) * 100, avg_train_loss / 100, avg_lat_acc / 100,
                        avg_lon_acc / 100, avg_train_time / 100))

            # set the zero
            avg_train_loss = 0
            avg_train_time = 0
            avg_lat_acc = 0
            avg_lon_acc = 0

    # validation the model
    net.eval()
    with torch.no_grad():
        print("**********************" + 'Validation Epoch:',
              str(epoch) + "**********************")
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0

        for i, data in enumerate(valDataloader):
            start_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, _, _, cls, nbrscls, _, _, temporal_mask = data

            # Move the data to GPU
            if args.use_cuda:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()
                va = va.cuda()
                nbrsva = nbrsva.cuda()
                lane = lane.cuda()
                nbrslane = nbrslane.cuda()
                cls = cls.cuda()
                nbrscls = nbrscls.cuda()

            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask)
            loss_mse, index, b_index = detr_test_mse(fut_pred, fut, op_mask)

            # Calculate the accuracy
            avg_val_lat_acc += torch.sum(
                torch.argmax(lat_pred[index, b_index], dim=1) == torch.argmax(lat_enc, dim=1)).item() / args.batch_size
            avg_val_lon_acc += torch.sum(
                torch.argmax(lon_pred[index, b_index], dim=1) == torch.argmax(lon_enc, dim=1)).item() / args.batch_size

            avg_val_loss += loss_mse.item()
            val_batch_count += 1

        print('Validation is done!')
        print('Validation Epoch: {}, Avg val loss: {:.3f}, Lat_Acc: {:.3f}, Lon_Acc: {:.3f}, '.format(
            epoch, avg_val_loss / val_batch_count, avg_val_lat_acc / val_batch_count,
            avg_val_lon_acc / val_batch_count))

        # save the model
        if args.network == 'highwaynet':
            if args.dataset == 'ngsim':
                save_path = f'checkpoints/ngsim/'
            elif args.dataset == 'highd':
                save_path = f'checkpoints/highd/'
            else:
                raise ValueError("Unsupported dataset!")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'epoch_{epoch}.pth')
        torch.save(net.state_dict(), save_path)