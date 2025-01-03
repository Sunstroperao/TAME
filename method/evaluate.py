import torch
import os
from highwaynet import HighwayNet
from utils import detr_val_mse
from config import get_args_parser
from ngsim_dataset import NgsimDataset
from torch.utils.data import DataLoader


args = get_args_parser().parse_args()  # get the args of HighwayNet

# Set the GPU
CUDA_VISIBLE_DEVICES = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

# Set the args
args.network = 'highwaynet'
args.dataset = 'ngsim'
args.input_dim = 6


if args.network == 'highwaynet':
    net = HighwayNet(args)
    net.load_state_dict(torch.load('checkpoints/{}/best_weight.pth'.format(args.dataset)))

if args.use_cuda:
    net = net.cuda()

# Initialize the dataset
root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get the root path
if args.dataset == 'ngsim':
    tsSet_path = os.path.join(root_path, 'data/ngsimdata/TestSet.mat')
elif args.dataset == 'highd':
    tsSet_path = os.path.join(root_path, 'data/highddata/TestSet.mat')
tsSet = NgsimDataset(tsSet_path)
tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=False, num_workers=args.num_workers, collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(25).cuda()
lossVals_de = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()
count = 0
avg_lat_acc = 0
avg_lon_acc = 0
net.eval()
with torch.no_grad():
    for i, data in enumerate(tsDataloader):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, _, _, cls, nbrscls, _, nbrs_num_batch,temporal_mask = data

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

        fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls,temporal_mask)
        l_rmse, l_de, c, idx, idx_b = detr_val_mse(fut_pred, fut, op_mask)

        lossVals += l_rmse.detach()
        lossVals_de += l_de.detach()
        counts += c.detach()
        count += 1

        avg_lat_acc += torch.sum(
            torch.argmax(lat_pred[idx, idx_b], dim=1) == torch.argmax(lat_enc, dim=1)).item() / 128
        avg_lon_acc += torch.sum(
            torch.argmax(lon_pred[idx, idx_b], dim=1) == torch.argmax(lon_enc, dim=1)).item() / 128



        if count % 100 == 0:
            rmse = torch.pow(lossVals / counts, 0.5) * 0.3048
            fde =(lossVals_de / counts) * 0.3048
            print('Test Iteration: {} | lat acc: {:.4f} | lon acc: {:.4f}'.format(count, avg_lat_acc / count, avg_lon_acc / count))
            print('Test RMSE: ', '1s: {:.2f} | 2s: {:.2f} | 3s: {:.2f} | 4s: {:.2f} | 5s: {:.2f}'.format(rmse[4], rmse[9], rmse[14], rmse[19], rmse[24]))
            print('Test FDE:  ', '1s: {:.2f} | 2s: {:.2f} | 3s: {:.2f} | 4s: {:.2f} | 5s: {:.2f}'.format(fde[4], fde[9], fde[14], fde[19], fde[24]))
            print('Test ADE:  ', '1s: {:.2f} | 2s: {:.2f} | 3s: {:.2f} | 4s: {:.2f} | 5s: {:.2f}'.format(
                torch.sum(lossVals_de[:4]) / torch.sum(counts[:4]) * 0.3048,
                torch.sum(lossVals_de[:9]) / torch.sum(counts[:9]) * 0.3048,
                torch.sum(lossVals_de[:14]) / torch.sum(counts[:14]) * 0.3048,
                torch.sum(lossVals_de[:19]) / torch.sum(counts[:19]) * 0.3048,
                torch.sum(lossVals_de[:24]) / torch.sum(counts[:24]) * 0.3048))
            print('==================================================================')

