# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @File Name： train.py
# @Time     :  2023/5/30
# @Author   :  Jiang Hao
# @Mail     :  jianghaotbs@163.com
import os
import time

import torch.nn as nn
from highwaynet import HighwayNet
from detr import DetrModel
import torch
import sys
sys.path.append('..')
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import wandb
from torch.utils.data import DataLoader
from utils import detr_loss, detr_test_mse, seed_everything
from ngsim_dataset import NgsimDataset
from config import get_args_parser, get_args_detr
# from tensorboardX import SummaryWriter

args = get_args_parser().parse_args()  # get the args of HighwayNet
# args = get_args_detr().parse_args()  # get the args of DetrModel

# Set the args
args.network = 'highwaynet'
args.dataset = 'ngsim'
args.batch_size = 512
args.input_dim = 6



# Set the seed
# seed_everything(seed=123456)

# Initialize the network
if args.network == 'highwaynet':
    net = HighwayNet(args)  # initialize the HighwayNet
elif args.network == 'detr':
    net = DetrModel(args)  # initialize the DetrModel


args.local_rank = int(os.environ['LOCAL_RANK'])
args.world_size = int(os.environ['WORLD_SIZE'])
args.rank = int(os.environ['RANK'])
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl')
# dist.barrier()

device = torch.device(args.local_rank)


# print the number of parameters
if args.local_rank == 0:    
    print('参数数目：', sum(param.numel() for param in net.parameters()))
if args.use_cuda:
    net = net.to(device)
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)


# visualization on tensorboard
# if args.local_rank == 0:
    # writer = SummaryWriter('logs/manet_{}_{}_ddp'.format(args.dataset, args.input_dim)) 
    # wandb.init(project='MQNet', name='Detr_{}'.format(args.input_dim), config={"dataset": args.dataset,
                                                                                                    #  "input_dim": args.input_dim,
                                                                                                    #  "model": args.network,})


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

trSampler = torch.utils.data.distributed.DistributedSampler(trSet)
valSampler = torch.utils.data.distributed.DistributedSampler(valSet)


trDataloader = DataLoader(trSet, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=trSet.collate_fn, sampler=trSampler)
valDataloader = DataLoader(valSet, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=valSet.collate_fn, sampler=valSampler)


# Start training and validation
step = 0
best_val_loss = 100
for epoch in range(args.epochs):
    trSampler.set_epoch(epoch)

    if args.local_rank == 0:
        # trDataloader = tqdm(trDataloader)
        print("**********************" + 'Training Epoch:', str(epoch+1) + "**********************")
    avg_train_loss = 0
    avg_train_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    net.train()
    for i, data in enumerate(tqdm(trDataloader)):
        start_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, _, _, cls, nbrscls, _, _,temporal_mask = data

        # Move the data to GPU
        if args.use_cuda:
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut.to(device)
            op_mask = op_mask.to(device)
            va = va.to(device)
            nbrsva = nbrsva.to(device)
            lane = lane.to(device)
            nbrslane = nbrslane.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)
            temporal_mask = temporal_mask.to(device)

        optimizer.zero_grad()
        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask)
        loss_mse, index, b_index = detr_loss(fut_pred, fut, op_mask)
        loss = loss_mse + cross_entropy(lat_pred[index, b_index], lat_enc) + cross_entropy(lon_pred[index, b_index], lon_enc)
        

        loss.backward()

        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # gradient clipping only used in HighwayNet
        optimizer.step()
        
        dist.all_reduce(loss)
        loss = loss / args.world_size   

        batch_time = time.time() - start_time  # each batch_time
        avg_train_time += batch_time  # total batch_time
        avg_train_loss += loss.item()  # total loss

        # Calculate the accuracy
        avg_lat_acc += torch.sum(
            torch.argmax(lat_pred[index, b_index], dim=1) == torch.argmax(lat_enc, dim=1)).item() / args.batch_size
        avg_lon_acc += torch.sum(
            torch.argmax(lon_pred[index, b_index], dim=1) == torch.argmax(lon_enc, dim=1)).item() / args.batch_size

        # print the loss and accuracy every 100 iterations
        if (i + 1) % 100 == 0 and args.rank == 0:
            print('Epoch: {}, Iteration: {:.2f}, Avg train loss: {:.3f}, Lat_Acc: {:.3f}, Lon_Acc: {:.3f}, Time: {:.3f}'
                .format(epoch+1, i / (len(trSet) / args.batch_size) * 100 * 4, avg_train_loss / 100, avg_lat_acc / 100,
                        avg_lon_acc / 100, avg_train_time / 100))
            
            # visualization tensorboard
            # writer.add_scalar('Train_Loss', avg_train_loss / 100, step)
            # writer.add_scalars('Train_Acc', {'Lat_Acc': avg_lat_acc / 100, 'Lon_Acc': avg_lon_acc / 100}, step)
            # wandb.log({"Train_Loss": avg_train_loss / 100, "Lat_Acc": avg_lat_acc / 100, "Lon_Acc": avg_lon_acc / 100, "step": step})
            step += 1

            # set the zero
            avg_train_loss = 0
            avg_train_time = 0
            avg_lat_acc = 0
            avg_lon_acc = 0

    # validation the model
    net.eval()
    with torch.no_grad():
        if args.local_rank == 0:
            valDataloader = tqdm(valDataloader)
            print("**********************" + 'Validation Epoch:', str(epoch+1) + "**********************")
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0

        for i, data in enumerate(valDataloader):
            start_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, _, _, cls, nbrscls, _, _, temporal_mask = data

        # Move the data to GPU
            if args.use_cuda:
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut.to(device)
                op_mask = op_mask.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                temporal_mask = temporal_mask.to(device)


            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask)
            loss_mse, index, b_index = detr_test_mse(fut_pred, fut, op_mask)

            # Calculate the accuracy
            avg_val_lat_acc += torch.sum(
                torch.argmax(lat_pred[index, b_index], dim=1) == torch.argmax(lat_enc, dim=1)).item() / args.batch_size
            avg_val_lon_acc += torch.sum(
                torch.argmax(lon_pred[index, b_index], dim=1) == torch.argmax(lon_enc, dim=1)).item() / args.batch_size

            dist.all_reduce(loss_mse)
            loss_mse = loss_mse / args.world_size

            avg_val_loss += loss_mse.item()
            val_batch_count += 1
        

        
        if args.local_rank == 0:
            print('Validation is done!')
            print('Validation Epoch: {}, Avg val loss: {:.3f}, Lat_Acc: {:.3f}, Lon_Acc: {:.3f}, '.format(
                epoch+1, avg_val_loss / val_batch_count, avg_val_lat_acc / val_batch_count,
                avg_val_lon_acc / val_batch_count))
            
            # visualization tensorboard
            # writer.add_scalar('Val_Loss', avg_val_loss / val_batch_count, epoch)
            # writer.add_scalars('Val_Acc', {'Lat_Acc': avg_val_lat_acc / val_batch_count, 'Lon_Acc': avg_val_lon_acc / val_batch_count}, epoch)
            # wandb.log({"Val_Loss": avg_val_loss / val_batch_count, "VLat_Acc": avg_val_lat_acc / val_batch_count, "VLon_Acc": avg_val_lon_acc / val_batch_count, "epoch": epoch})

            # save the model
            if avg_val_loss / val_batch_count < best_val_loss:
                best_val_loss = avg_val_loss / val_batch_count
                print('\033[1;35mSaving the model...\033[0m')
                torch.save(net.module.state_dict(), 'trained_model/{}/{}/best_weight.pth'.format(args.network, args.dataset))
                print('\033[1;35mThe model is saved successfully!\033[0m')

if args.local_rank == 0:
    # writer.close()
    # wandb.finish()
    print('Training is done!')