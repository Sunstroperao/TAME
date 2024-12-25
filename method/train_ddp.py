import os
import time

import torch.nn as nn
from highwaynet import HighwayNet
import torch
import sys
sys.path.append('..')
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from utils import detr_loss, detr_test_mse, seed_everything
from ngsim_dataset import NgsimDataset
from config import get_args_parser


args = get_args_parser().parse_args()  # get the args of HighwayNet

# Set the args
args.network = 'highwaynet'
args.dataset = 'highd'
args.batch_size = 512
args.input_dim = 6



# Initialize the network
if args.network == 'highwaynet':
    net = HighwayNet(args)  # initialize the HighwayNet


args.local_rank = int(os.environ['LOCAL_RANK'])
args.world_size = int(os.environ['WORLD_SIZE'])
args.rank = int(os.environ['RANK'])
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl')
# dist.barrier()

device = torch.device(args.local_rank)

if args.use_cuda:
    net = net.to(device)
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)


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
best_val_loss = 100
for epoch in range(args.epochs):
    trSampler.set_epoch(epoch)

    if args.local_rank == 0:
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
            

            # save the model
            if avg_val_loss / val_batch_count < best_val_loss:
                best_val_loss = avg_val_loss / val_batch_count
                print('\033[1;35mSaving the model...\033[0m')
                save_path = f'checkpoints/{args.dataset}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(net.module.state_dict(), os.path.join(save_path, 'best_weight.pth'))
                print('\033[1;35mThe model is saved successfully!\033[0m')

if args.local_rank == 0:
    print('Training is done!')