#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import pdb
import random
import argparse
import time
import warnings
import matplotlib

matplotlib.use('Agg')
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
parser.add_argument('--subj-train', default="/data/guotao/synthrad2023_brain/train_subj_v2.pkl",
                    help='subjects used for train')
parser.add_argument('--subj-val', default="/data/guotao/synthrad2023_brain/val_subj_v2.pkl",
                    help='subjects used for validation')
parser.add_argument('--subj-test', default="/data/guotao/synthrad2023_brain/test_subj_v2.pkl",
                    help='subjects used for test')
parser.add_argument('--vol-path', default="/data/guotao/synthrad2023_brain/seg/volumes_center/",
                    help='path to cross modality volume')
parser.add_argument('--seg-path', default="/data/guotao/synthrad2023_brain/seg/seg_center/",
                    help='path to cross modality segmentation')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--mode', default='mr>ct', help='register from mr -> ct')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--scale', type=float, default=1.0, help='scale factor of the original volume')
parser.add_argument('--chunk', action='store_true', help='whether to use chunk the volumes')

# training parameters
parser.add_argument('--gpu', default=None, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=1000,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--load-model-dds', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--warm-up', type=float, default=0, help='rate of warm up epochs')
parser.add_argument('--no-amp', action='store_true', help='NOT auto mix precision training')

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
parser.add_argument('--model', type=str, default='vm', help='Choose a model to train (vm, vm-feat, mm, mm-feat)')

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

@contextmanager
def conditional_autocast(enabled: bool):
    if enabled:
        with autocast():
            yield
    else:
        yield

# load and prepare data
with open(args.subj_train, 'rb') as file:
    train_subject = pickle.load(file)
with open(args.subj_val, 'rb') as file:
    val_subject = pickle.load(file)
with open(args.subj_test, 'rb') as file:
    test_subject = pickle.load(file)

data_example = vxm.py.utils.load_volfile(
    args.seg_path + train_subject[0] + '.nii.gz')
inshape = tuple([int(old_size * args.scale) for old_size in data_example.shape])
labels_in = np.unique(data_example)

if args.chunk:
    assert args.scale == 0.5, "If using chunking opearation, the scale must be 0.5!"
    args.scale *= 2

generator = src_generators.volgen_crossmodality(
    subjects=train_subject,
    vol_path=args.vol_path,
    seg_path=args.seg_path,
    labels=labels_in,
    mode=args.mode,
    same_subject=False,
    batch_size=args.batch_size,
    resize_factor=args.scale,
    chunk=args.chunk,
)

generator_val = src_generators.volgen_crossmodality(
    subjects=val_subject,
    vol_path=args.vol_path,
    seg_path=args.seg_path,
    labels=labels_in,
    mode=args.mode,
    same_subject=False,
    batch_size=args.batch_size,
    resize_factor=args.scale,
    chunk=args.chunk,
)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# prepare model folder
model_dir = args.model_dir
if os.path.exists(model_dir):
    warnings.warn("Ensure that you don't overwrite the former folder!")
# assert not os.path.exists(model_dir), "Ensure that you don't overwrite the former folder!"
os.makedirs(model_dir, exist_ok=True)

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

# unet architecture
enc_nf = args.enc if args.enc else [16] * 4
dec_nf = args.dec if args.dec else [16] * 6

# Define a model
if args.model == 'vm':
    model = networks.VxmDense.load(args.load_model, device) \
        if args.load_model else \
        networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
        )
elif args.model == 'vm-feat':
    model = networks.VxmFeat.load(args.load_model, device) \
        if args.load_model else \
        networks.VxmFeat(
            inshape=inshape,
            nb_feat_extractor=[[16] * 2, [16] * 4],
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
        )
elif args.model == 'mm':
    config = CONFIGS_TM['MambaMorph']
    model = TransMorph.MambaMorph(config)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
elif args.model == 'mm-feat':
    config = CONFIGS_TM['MambaMorph']
    model = TransMorph.MambaMorphFeat(config)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
transform_model = SpatialTransformer(inshape, mode='bilinear')  # STN
transform_model.to(device)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = src_loss.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = src_loss.MSE().loss
elif args.image_loss == 'dice':
    image_loss_func = src_loss.Dice().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss (regularization loss)
losses += [src_loss.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]  # Regularization loss

if args.cl > 0.:
    cl_loss_fn = src_loss.ContrastiveSem(scale=0.5)
    cl_weight = args.cl

min_loss = 1e3
train_rec_path = os.path.join(model_dir, 'train_record.txt')
val_rec_path = os.path.join(model_dir, 'val_record.txt')
if os.path.exists(train_rec_path):
    os.remove(train_rec_path)
if os.path.exists(val_rec_path):
    os.remove(val_rec_path)
# TODO: Set hyperparameter
# lr, eps = args.lr, 1e-3
lr, eps = args.lr, 1e-3
r_scale = 5.
reg_weight = 0.001
lowest_loss = 1e3

# TODO: set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if not args.no_amp:
    scaler = torch.cuda.amp.GradScaler()

# set the ignorable labels
if len(args.ignore_label) > 0:
    label_ignore = ~np.isin(labels_in, args.ignore_label)
else:
    label_ignore = np.ones(len(labels_in)).bool()
label_ignore = torch.from_numpy(label_ignore).to(device)

total_step = math.ceil(len(train_subject) / args.batch_size)
if args.chunk:
    args.batch_size *= 8

train_loss_rec = []
val_loss_rec = []
cl_loss_rec = []

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    print(f"Epoch {epoch + 1} / {args.epochs}")
    epoch_loss = []
    epoch_total_loss = []
    step_start_time = time.time()
    train_rec = open(train_rec_path, 'a')
    val_rec = open(os.path.join(model_dir, 'val_record.txt'), 'a')

    model.train()
    if args.cl > 0:
        cl_epoch_rec = []

    for step in range(total_step):
        vols, segs = next(generator)
        zero_flow = np.zeros(
            (args.batch_size, *tuple([int(tmp / args.int_downsize) for tmp in inshape]), len(inshape)))
        inputs = [vols[0], vols[1]]  # src_img, tgt_img
        y_true = [segs[1], zero_flow]  # tgt_label, 0
        src_label = segs[0]
        src_label = torch.from_numpy(src_label).to(device).float().permute(0, 4, 1, 2, 3)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # volume pairs
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

        with conditional_autocast(not args.no_amp):
            ret_dict = model(*inputs, return_pos_flow=True, return_feature=True)
            warped_vol = ret_dict['moved_vol']
            preint_flow = ret_dict['preint_flow']
            pos_flow = ret_dict['pos_flow']
            # warped_vol, preint_flow, pos_flow = model(*inputs, return_both_flow=True)
            warped_label = transform_model(src_label, pos_flow)
            y_pred = (warped_label, preint_flow)
            loss = 0
            loss_list = []
            per_loss = torch.zeros([args.batch_size], device=device)
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n], ignore_label=label_ignore)
                curr_loss *= weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            # print(f"Step: {step}, Loss: {loss}")
            if args.cl > 0:
                feat = ret_dict['feature']
                cl_loss = cl_loss_fn.loss(feat, src_label, y_true[0], ignore_label=label_ignore)
                # print(f"Step: {step}, Loss: {loss}, CL_loss: {cl_loss}")
                loss += cl_loss * cl_weight
                cl_epoch_rec.append(cl_loss.detach().cpu().item())

        optimizer.zero_grad()
        if not args.no_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)  # total_loss, sim loss, reg loss
    print(' - '.join((epoch_info, loss_info)), flush=True)
    train_rec.write(f"Epoch {epoch + 1} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if args.cl > 0:
        cl_loss_rec.append(np.mean(cl_epoch_rec))
        train_rec.write(f"{round(np.mean(epoch_total_loss), 6)}, {losses_info}, {round(np.mean(cl_epoch_rec), 6)}\n")
    else:
        train_rec.write(f"{round(np.mean(epoch_total_loss), 6)}, {losses_info}\n")
    train_loss_rec.append(np.mean(epoch_total_loss))

    # save model checkpoint
    if epoch % args.steps_per_epoch == 0 and epoch > 0:
        if args.model.startswith('vm'):
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % epoch))

    if np.mean(epoch_total_loss) < lowest_loss:
        if args.model.startswith('vm'):
            model.save(os.path.join(model_dir, 'min_train.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, 'min_train.pt'))
        lowest_loss = np.mean(epoch_total_loss)

    # validating loops
    with torch.no_grad():
        model.eval()
        val_total_loss = []
        val_loss = []

        for val_step in range(math.ceil(len(val_subject) / args.batch_size)):
            # generate inputs (and true outputs) and convert them to tensors
            vols, segs = next(generator_val)
            inputs = [vols[0], vols[1]]  # src_img, tgt_img
            y_true = [segs[1], zero_flow]  # tgt_label, 0
            src_label = segs[0]
            src_label = torch.from_numpy(src_label).to(device).float().permute(0, 4, 1, 2, 3)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # volume pairs
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            with conditional_autocast(not args.no_amp):
                ret_dict = model(*inputs, return_pos_flow=True)
                warped_vol = ret_dict['moved_vol']
                preint_flow = ret_dict['preint_flow']
                pos_flow = ret_dict['pos_flow']
                warped_label = transform_model(src_label, pos_flow)
                y_pred = (warped_label, preint_flow)
                # calculate total loss
                loss_val = 0
                loss_list_val = []
                for n, loss_function in enumerate(losses):
                    curr_loss = loss_function(y_true[n], y_pred[n], ignore_label=label_ignore) * weights[n]
                    loss_list_val.append(curr_loss.item())
                    loss_val += curr_loss
                val_loss.append(loss_list_val)
                val_total_loss.append(loss_val.item())

        val_losses_info = ', '.join(['%.4e' % f for f in np.mean(val_loss, axis=0)])
        print(f"---Validation Loss: {round(np.mean(val_total_loss), 6)} "
              f"({val_losses_info})")
        mean_loss = round(np.mean(val_total_loss), 6)
        val_rec.write(f"Epoch {epoch + 1} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        val_rec.write(f"{mean_loss}, {val_losses_info}\n")
        val_loss_rec.append(mean_loss)

        if np.mean(val_total_loss) < min_loss:
            if args.model.startswith('vm'):
                model.save(os.path.join(model_dir, 'min_val.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, 'min_val.pt'))
            min_loss = np.mean(val_total_loss)

    time_consumption = round((time.time() - step_start_time) / 60, 3)
    print('-' * 5 + f'This epoch takes {time_consumption} min.\n')
    train_rec.close()
    val_rec.close()

print(f"Minimum training loss is: {lowest_loss}.")

# plot the loss
plt.plot(np.array(train_loss_rec))
plt.title('Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(model_dir, 'train_loss.png'))
plt.close()

plt.plot(np.array(val_loss_rec))
plt.title('Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(model_dir, 'val_loss.png'))
plt.close()

if args.cl > 0:
    plt.plot(np.array(cl_loss_rec))
    plt.title('Contrastive loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_dir, 'cl_loss.png'))
    plt.close()

# final model save
if args.model.startswith('vm'):
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
else:
    torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % epoch))

print('-' * 75)
print('-' * 30 + "Finish training" + '-' * 30)
print('-' * 75)
""""""
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
    # Prepare the model
    if args.model == 'vm':
        model = networks.VxmDense.load(os.path.join(model_dir, 'min_train.pt'), device)
    elif args.model == 'vm-feat':
        model = networks.VxmFeat.load(os.path.join(model_dir, 'min_train.pt'), device)
    elif args.model.startswith('mm'):
        model.load_state_dict(torch.load(os.path.join(model_dir, 'min_train.pt')))
    model.to(device)
    model.eval()
    transform_model = SpatialTransformer(inshape, mode='nearest')  # STN
    transform_model.to(device)
    dice_means = []
    with torch.no_grad():
        for idx in range(len(pairings)):
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
            # moved, warp = model(input_moving, input_fixed, registration=True)
            ret_dict = model(input_moving, input_fixed, return_pos_flow=True)
            moved = ret_dict['moved_vol']
            warp = ret_dict['pos_flow']

            # apply transform
            input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
            warped_seg = transform_model(input_seg, warp).squeeze()
            # compute volume overlap (dice)
            overlap = vxm.py.utils.dice(np.argmax(warped_seg.cpu().numpy(), axis=0),
                                        np.argmax(fixed_seg, axis=-1).squeeze(), include_zero=True)
            dice_means.append(np.mean(overlap[label_ignore.cpu().numpy()]))

    print('**TEST: Avg Dice %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))
    with open(os.path.join(model_dir, 'val_record.txt'), 'a') as test_rec:
        test_rec.write(f"TEST   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        test_rec.write(f"{round(np.mean(dice_means), 6)} +/- {round(np.std(dice_means), 6)}\n")
