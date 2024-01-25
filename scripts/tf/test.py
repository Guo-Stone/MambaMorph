#!/usr/bin/env python

"""
Example script for testing quality of trained VxmDense models. This script iterates over a list of
images pairs, registers them, propagates segmentations via the deformation, and computes the dice
overlap. Example usage is:

    test.py  \
        --model model.h5  \
        --pairs pairs.txt  \
        --img-suffix /img.nii.gz  \
        --seg-suffix /seg.nii.gz

Where pairs.txt is a text file with line-by-line space-seperated registration pairs.
This script will most likely need to be customized to fit your data.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

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
import argparse
import time
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import pdb
from scipy.ndimage import zoom
from voxelmorph.py.utils import jacobian_determinant as jd


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--model', required=True, help='VxmDense model file')
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# sanity check on input pairs
if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
    print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
    exit(1)
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load seg labels if provided
labels = np.load(args.labels) if args.labels else None

# check if multi-channel data
add_feat_axis = not args.multichannel

# keep track of all dice scores
reg_times = []
dice_means = []
RoI_dice = []
Jacobian_ratio = []

with tf.device(device):

    # load model and build nearest-neighbor transfer model
    model = vxm.networks.VxmDense.load(args.model, input_model=None)
    registration_model = model.get_registration_model()
    inshape = registration_model.inputs[0].shape[1:-1]
    transform_model = vxm.networks.Transform(inshape, interp_method='nearest')
    for i in range(len(img_pairs)):
        # load moving image and seg
        moving_vol = vxm.py.utils.load_volfile(
            img_pairs[i][0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][0], np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)

        # load fixed image and seg
        fixed_vol = vxm.py.utils.load_volfile(
            img_pairs[i][1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][1], np_var='seg')

        # TODO: transpose dimensions
        moving_vol = moving_vol.swapaxes(2, 3)
        moving_seg = moving_seg.swapaxes(2, 3)
        fixed_vol = fixed_vol.swapaxes(2, 3)
        fixed_seg = fixed_seg.swapaxes(1, 2)

        # TODO: resample the image to fit SynthMorph
        # new_shape = (1, 160, 160, 192, 1)  # For SynthMorph
        # zoom_factors = [new_size / float(old_size) for new_size, old_size in zip(new_shape, moving_vol.shape)]
        # moving_vol = zoom(moving_vol, zoom_factors, order=0)
        # moving_seg = zoom(moving_seg, zoom_factors, order=0)
        # fixed_vol = zoom(fixed_vol, zoom_factors, order=0)
        # fixed_seg = zoom(fixed_seg, zoom_factors[1:-1], order=0)

        # predict warp and time
        start = time.time()
        warp = registration_model.predict([moving_vol, fixed_vol])
        # TODO: Calculating |J| < 0
        for idx in range(len(warp)):
            jacobian_D = jd(warp[idx])
            vol_num = 1
            for num in jacobian_D.shape:
                vol_num *= num
            minus_ratio = (jacobian_D < 0).sum() / vol_num
            Jacobian_ratio.append(minus_ratio)

        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)
        # apply transform
        warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
        # compute volume overlap (dice)
        overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=labels)
        dice_means.append(np.mean(overlap))
        print('Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f'
              % (i + 1, reg_time, np.mean(overlap), np.std(overlap)))
        # TODO: To get specific anatomical dice score
        RoI_index = np.array([1, 2, 20, 21, 14, 30, 13])
        RoI_dice.append(overlap[RoI_index])


print()
print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                    np.std(reg_times)))
print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))

RoI_dice = np.array(RoI_dice)
print()
print(f'RoI inx are {RoI_index}')
print(f'RoI Dice mean: {RoI_dice.mean(axis=0)}')
print(f'RoI Dice std: {RoI_dice.std(axis=0)}')

print()
print(f'|J| < 0 mean: {np.mean(Jacobian_ratio)}')
