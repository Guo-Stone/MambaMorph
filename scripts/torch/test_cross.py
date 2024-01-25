#!/usr/bin/env python


import os
import pdb
import random
import argparse
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import math
import datetime
from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8, from packages instead of source code
from voxelmorph.torch.layers import SpatialTransformer
from voxelmorph.py.utils import jacobian_determinant as jd
import sys
import pickle

sys.path.append(r"/home/guotao/code/mambamorph-dev/mambamorph")
import generators as src_generators

sys.path.append(r"/home/guotao/code/mambamorph-dev/mambamorph/torch")
import losses as src_loss
import networks
import utils
from TransMorph import CONFIGS as CONFIGS_TM
import TransMorph as TransMorph

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--subj-test', default="/data/guotao/synthrad2023_brain/test_subj_v2.pkl",
                    help='subjects used for test')
parser.add_argument('--vol-path', default="/data/guotao/synthrad2023_brain/seg/volumes_center/",
                    help='path to cross modality volume')
parser.add_argument('--seg-path', default="/data/guotao/synthrad2023_brain/seg/seg_center/",
                    help='path to cross modality segmentation')
parser.add_argument('--load-model', required=True, help='optional model file to initialize with')
parser.add_argument('--model', type=str, default=None,
                    help='If you only load model params in load-model, you have to specify a model first')
parser.add_argument('--mode', default='mr>ct', help='register from mr -> ct')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--scale', type=float, default=1.0, help='scale factor of the original volume')
parser.add_argument('--chunk', action='store_true', help='whether to use chunk the volumes')

# training parameters
parser.add_argument('--gpu', default=None, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--feat', action='store_true', help='whether to use feature extractor before Registraion')

# loss hyperparameters
parser.add_argument('--image-loss', default='dice',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.1,
                    help='weight of deformation loss (default: 0.1)')
parser.add_argument('--ignore-label', type=int, nargs='+', default=[0, 5, 24],
                    help='list of ignorable labels')
parser.add_argument('--cl', type=float, default=0.0, help='whether to use contrastive loss and set its weight')
args = parser.parse_args()
bidir = args.bidir

# load and prepare data
with open(args.subj_test, 'rb') as file:
    test_subject = pickle.load(file)

data_example = vxm.py.utils.load_volfile(
    args.seg_path + test_subject[0] + '.nii.gz')
inshape = tuple([int(old_size * args.scale) for old_size in data_example.shape])
labels_in = np.unique(data_example)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# device handling
if args.gpu:
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
else:
    nb_gpus = 0
    device = 'cpu'

torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()

# Prepare a model
if args.model is None:
    model = torch.load(args.load_model)
elif args.model == 'vm':
    model = networks.VxmDense.load(args.load_model, device)
elif args.model == 'vm-feat':
    model = networks.VxmFeat.load(args.load_model, device)
elif args.model == 'vm-feat-double':
    model = networks.VxmFeatDouble.load(args.load_model, device)
elif args.model == 'vm-feat-fusion':
    model = networks.VxmFeatFusion.load(args.load_model, device)
elif args.model == 'tm':
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.load_state_dict(torch.load(args.load_model))
elif args.model == 'tm-feat':
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorphFeat(config)
    model.load_state_dict(torch.load(args.load_model))
elif args.model == 'mm':
    config = CONFIGS_TM['MambaMorph']
    model = TransMorph.MambaMorph(config)
    model.load_state_dict(torch.load(args.load_model))
elif args.model == 'mm-ori':
    config = CONFIGS_TM['MambaMorph']
    model = TransMorph.MambaMorphOri(config)
    model.load_state_dict(torch.load(args.load_model))
elif args.model == 'mm-feat':
    config = CONFIGS_TM['MambaMorph']
    model = TransMorph.MambaMorphFeat(config)
    model.load_state_dict(torch.load(args.load_model))

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)

# prepare the model for training and send to device
model.to(device)
total_params = sum([param.numel() for param in model.parameters()])
# set the ignorable labels
if len(args.ignore_label) > 0:
    label_ignore = ~np.isin(labels_in, args.ignore_label)
else:
    label_ignore = np.ones(len(labels_in)).bool()
label_ignore = torch.from_numpy(label_ignore).to(device)

# test loops
if test_subject is not None:
    np.random.seed(0)
    first_mod = args.mode.split('>')[0]
    second_mod = args.mode.split('>')[1]

    test_vol_moving = [os.path.join(args.vol_path, item + f"_{first_mod}.nii.gz") for item in test_subject]
    test_vol_fixed = [os.path.join(args.vol_path, item + f"_{second_mod}.nii.gz") for item in test_subject]
    test_seg = [os.path.join(args.seg_path, item + f".nii.gz") for item in test_subject]
    # Shuffle the test data
    numbers = np.arange(0, len(test_subject))
    random_numbers = np.random.permutation(numbers)
    pairings = np.split(random_numbers, len(random_numbers) // 2)
    model.eval()
    transform_model = SpatialTransformer(inshape, mode='nearest')  # STN
    transform_model.to(device)

    dice_means = []
    anatomical_dice = []
    Jacobian_ratio = []
    HD95 = []
    infer_time = []
    rec = 0
    with torch.no_grad():
        for idx in range(len(pairings)):
            print(f"We are testing {rec + 1} / {len(pairings)} sample...")
            rec += 1
            # read the images and labels from files
            moving = vxm.py.utils.load_volfile(test_vol_moving[pairings[idx][0]], add_batch_axis=True, np_var='vol',
                                               add_feat_axis=add_feat_axis, resize_factor=args.scale)
            moving = utils.minmax_norm(moving)
            moving_seg = vxm.py.utils.load_volfile(test_seg[pairings[idx][0]], add_batch_axis=True,
                                                   np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)

            fixed = vxm.py.utils.load_volfile(test_vol_fixed[pairings[idx][1]], add_batch_axis=True, np_var='vol',
                                              add_feat_axis=add_feat_axis, resize_factor=args.scale)
            fixed = utils.minmax_norm(fixed)
            fixed_seg = vxm.py.utils.load_volfile(test_seg[pairings[idx][1]], add_batch_axis=True,
                                                  np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)

            moving_seg = src_generators.split_seg_global(moving_seg, labels_in)
            fixed_seg = src_generators.split_seg_global(fixed_seg, labels_in)

            # Infer
            input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
            input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
            # predict
            start_time = time.time()
            ret_dict = model(input_moving, input_fixed)
            end_time = time.time()
            duration = end_time - start_time
            if idx == 0:
                memory_allocated = torch.cuda.max_memory_allocated()
                memory_cached = torch.cuda.max_memory_reserved()
                Gb_consumed = memory_cached / 1e9
            moved = ret_dict['moved_vol']
            if args.model.startswith('tm') or args.model == 'mm-ori':
                warp = ret_dict['preint_flow']
            else:
                warp = ret_dict['pos_flow']

            # Warp the moving segment
            input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
            warped_seg = transform_model(input_seg, warp).squeeze()

            # Dice
            overlap = vxm.py.utils.dice(np.argmax(warped_seg.cpu().numpy(), axis=0),
                                        np.argmax(fixed_seg, axis=-1).squeeze(), include_zero=True)
            dice_means.append(np.mean(overlap[label_ignore.cpu().numpy()] * 100))
            anatomical_dice.append(overlap[label_ignore.cpu().numpy()])

            # |J|<0
            minus_ratio = utils.negative_jacobin(warp[0].permute(1, 2, 3, 0).cpu().numpy())
            Jacobian_ratio.append(minus_ratio)

            # 95% Hausdorff distance
            warped_seg_np = warped_seg.cpu().permute(1, 2, 3, 0).numpy()
            fixed_seg_np = fixed_seg.squeeze()
            HD95_per_label = []
            for label_idx in np.where(~np.isin(labels_in, args.ignore_label))[0]:
                HD95_each = utils.hausdorff_distance(warped_seg_np[..., label_idx],
                                                     fixed_seg_np[..., label_idx], percentage=95)
                HD95_per_label.append(HD95_each)
            HD95.append(np.mean(HD95_per_label))
            # Calculating inference time
            if idx > 0:
                infer_time.append(duration)

    anatomical_dice = np.stack(anatomical_dice)
    anatomical_dice_mean = anatomical_dice.mean(axis=0)
    anatomical_dice_std = anatomical_dice.std(axis=0)

    print('Avg Dice:                %.2f +/- %.2f \n' % (np.mean(dice_means), np.std(dice_means)))
    print('|J| < 0 percentage:      %.6f +/- %.6f \n' % (np.mean(Jacobian_ratio), np.std(Jacobian_ratio)))
    print('HD95:                    %.2f +/- %.2f \n' % (np.mean(HD95), np.std(HD95)))
    print('Inference time:          %.4f +/- %.4f s\n' % (np.mean(infer_time), np.std(infer_time)))
    print('Memory occupation:       %.4f Gb\n' % Gb_consumed)
    print(f'Amounts of parameter:   {total_params / 1e6} Mb\n')
    print('Anatomical Dice (mean and std): \n')
    print(anatomical_dice_mean)
    print(anatomical_dice_std)
