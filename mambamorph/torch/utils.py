import torch
import os
import numpy as np
import pdb
from voxelmorph.torch.layers import SpatialTransformer
import cv2
from matplotlib import pyplot as plt
import numpy as np
import neurite as ne
import pystrum
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity, mean_squared_error

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8


def SGD(model, lr=0.01):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
        # model.zero_grad()


def pseudo_theta(model, eps=1e-3):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data += eps * param.grad


def save_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
            # if torch.isnan(param.grad).sum() > 0:
            #     return None
    return grads


def save_grads_v1(model, flatten=False):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # grads[name] = param.grad.clone().to(torch.float16)
            grads[name] = param.grad.clone()
    if not flatten:
        return grads
    else:
        grad_tensors = list(grads.values())
        flat_grad = torch.cat([grad.view(-1) for grad in grad_tensors])
        return flat_grad


def load_and_update_grads(model, grads: dict, lr=1):
    for name, param in model.named_parameters():
        param.data += lr * grads[name]


def load_grads(model, grads, grads_weights=None, add=False):
    if grads_weights is None:  # grads is dict
        for name, param in model.named_parameters():
            if name in grads.keys():
                if add:
                    param.grad += grads[name]
                else:
                    param.grad = grads[name].clone()
    else:  # grads is list
        assert len(grads) == len(grads_weights)
        for name, param in model.named_parameters():
            for idx in range(len(grads_weights)):
                param.grad += grads[idx][name] * grads_weights[idx]


def cal_r_i(grads_train: list, grads_val):
    num = len(grads_train)
    r_i = torch.zeros([num])
    if type(grads_val) is dict:
        grad_tensors = list(grads_val.values())
        flat_grad = torch.cat([grad.view(-1) for grad in grad_tensors])
        grads_val = flat_grad
    for idx in range(num):
        grad_tensors = list(grads_train[idx].values())
        flat_grad = torch.cat([grad.view(-1) for grad in grad_tensors])
        r_i[idx] = torch.nn.functional.cosine_similarity(grads_val, flat_grad, dim=0)
    return r_i


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    if arg[0].endswith('.txt'):
        with open(arg[0], 'r') as f:
            content = f.readlines()
        files = [f.strip() for f in content]
    else:
        import glob
        ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
        files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
        files = sum((glob.glob(f) for f in files), [])
        files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(vxm.py.utils.load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def minmax_norm(x, axis=None):
    """
    Min-max normalize array using a safe division.

    Arguments:
        x: Array to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized array.
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.divide(x - x_min, x_max - x_min, out=np.zeros_like(x - x_min), where=x_max != x_min)


def save_feature_map(feat_slice: np.array, save_path='feat_img.png'):
    # feat_slice: (3, H, W)
    channel, H, W = feat_slice.shape
    if channel > 3:
        # PCA
        pass
    assert channel == 3

    feat_map = np.zeros([H, W, channel], dtype='uint8')
    # Normalize to [0, 255]
    for idx in range(channel):
        feat_map[:, :, idx] = int(((feat_slice[idx] - np.min(feat_slice[idx])) /
                                   (np.max(feat_slice[idx]) - np.min(feat_slice[idx]))) * 255)
    bgr_image_array = cv2.cvtColor(feat_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_image_array)


def plot_loss(txt_path: str):
    loss = []
    with open(txt_path, 'r') as f:
        con = f.readlines()
    for line in con:
        if not line.startswith('Epoch'):
            loss.append(float(line.split(',')[0]))
    fig, ax = plt.subplots()
    ax.plot(loss, label='loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('Loss.png')


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype('bool'))
    reference = np.atleast_1d(reference.astype('bool'))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hausdorff_distance(result, reference, voxelspacing=None, connectivity=1, percentage=None):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    if percentage is None:
        distance = max(hd1.max(), hd2.max())
    elif isinstance(percentage, (int, float)):
        distance = np.percentile(np.hstack((hd1, hd2)), percentage)
    else:
        raise ValueError
    return distance


def negative_jacobin(flow):
    """
    flow: numpy.array, [W, H, L, 3]
    """
    w, h, l, c = np.shape(flow)

    flow_image = sitk.GetImageFromArray(flow.astype('float64'), isVector=True)
    determinant = sitk.DisplacementFieldJacobianDeterminant(flow_image)
    neg_jacobin = (sitk.GetArrayFromImage(determinant)) < 0
    cnt = np.sum(neg_jacobin)
    norm_cnt = cnt / (h * w * l)
    return norm_cnt * 100


def plot_deformation(warp):
    volshape = warp.shape[2:]
    grid = pystrum.pynd.ndutils.bw_grid(vol_shape=volshape, spacing=11)
    transform_model = SpatialTransformer(volshape)  # STN, bilinear
    transform_model.to(warp.device)
    input_grid = torch.from_numpy(grid[None, ..., None]).to(warp.device).float().permute(0, 4, 1, 2, 3)
    warped_grid = transform_model(input_grid, warp).squeeze().cpu().numpy()
    ne.plot.slices(warped_grid[:,:,volshape[-1]//2], width=3)
    plt.savefig('tmp.png')
    plt.close()