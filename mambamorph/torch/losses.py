import pdb
import voxelmorph as vxm
import torch
import torch.nn.functional as F
import numpy as np
import math

cl_cfg = dict(
    pre_select_pos_number=10000,  # default 2000
    after_select_pos_number=100,  # default 100
    pre_select_neg_number=1000,  # default 2000
    after_select_neg_number=250,  # default 500
    positive_distance=0.95,  # default 2.
    ignore_distance=40.,
    coarse_positive_distance=25.,
    coarse_ignore_distance=5.,
    coarse_z_thres=6.,
    coarse_pre_select_neg_number=250,
    coarse_after_select_neg_number=200,
    fine_temperature=0.25,  # default 0.5
    coarse_temperature=0.5,
    select_pos_num=1000,
    select_neg_num=5000, )


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        if weight is not None:
            B = len(cc)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=cc.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=cc.device)
            for idx in range(B):
                item_loss = -torch.mean(cc[idx])
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False):
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=y_true.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=y_true.device)
            for idx in range(B):
                item_loss = torch.mean((y_true[idx] - y_pred[idx]) ** 2)
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=y_true.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=y_true.device)
            for idx in range(B):
                top = 2 * (y_true[idx:idx + 1] * y_pred[idx:idx + 1]).sum(dim=vol_axes)
                bottom = torch.clamp((y_true[idx:idx + 1] + y_pred[idx:idx + 1]).sum(dim=vol_axes), min=1e-5)
                if ignore_label is not None:
                    item_dice = -torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
                else:
                    item_dice = -torch.mean(top / bottom)
                weighted_loss += item_dice * weight[idx]
                per_loss[idx] = item_dice
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            top = 2 * (y_true * y_pred).sum(dim=vol_axes)
            bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
            if ignore_label is not None:
                dice = torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
            else:
                dice = torch.mean(top / bottom)
        return -dice

    def each_dice(self, y_true, y_pred, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        if ignore_label is not None:
            dice = top[:, ignore_label] / bottom[:, ignore_label]
        else:
            dice = top / bottom
        return dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]
        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        if weight is not None:
            B = len(grad)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=grad.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=grad.device)
            for idx in range(B):
                weighted_loss += grad[idx] * weight[idx]
                per_loss[idx] = grad[idx]
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return grad.mean()


def meshgrid3d(inshape):
    z_ = torch.linspace(0., inshape[0] - 1, inshape[0])
    y_ = torch.linspace(0., inshape[1] - 1, inshape[1])
    x_ = torch.linspace(0., inshape[2] - 1, inshape[2])
    z, y, x = torch.meshgrid(z_, y_, x_)
    return torch.stack((z, y, x), 3)


class ContrastivePos:
    def __init__(self, scale=1., norm=True):
        """
        scale: If you need to calculate CL loss within cropped volumes, set scale < 1.
        norm: Set True and it will normalize input feature
        """
        super().__init__()
        assert scale <= 1, "The parameter 'scale' mustn't be more than 1."
        self.scale = scale
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def single_contrastive_loss(self, feat, mask):
        """
        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        mask: (1, 1, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        mesh = meshgrid3d(mask.squeeze().shape).to(s_feat.device)  # (H, W, L, 3)

        # Get the foregound points (index) through mask
        foregrond_points = mesh[mask[0, 0, :, :, :] == 1, :]  # (num_points_in_mask, 3)

        # Select pre-positive points, 'points_select' stores the index of selected points
        points_select = torch.randperm(foregrond_points.shape[0], device=s_feat.device)[
                        :cl_cfg["pre_select_pos_number"]]  # (num_pre_select_points)

        # Record the pre-positive points' location
        select_points = foregrond_points[points_select, :].transpose(0, 1)  # (3, num_pre_select_points)

        with torch.no_grad():
            dist = torch.linalg.norm((select_points.view(3, select_points.shape[1], 1)
                                      - foregrond_points.transpose(0, 1)[:, None, :]), dim=0)

        # Get positive points through distance
        pos_match = torch.unique(torch.where(dist < cl_cfg['positive_distance'])[0])

        # Filter the positive points again
        if pos_match.shape[0] == 0:
            pos_match = torch.unique(torch.where(dist < dist.min() + 0.5)[0])
        if pos_match.shape[0] <= cl_cfg['after_select_pos_number']:
            pos_match = pos_match
        else:
            pos_match = pos_match[torch.randperm(pos_match.shape[0])[:cl_cfg['after_select_pos_number']]]

        # Fix a set of points, record their distance to other points
        dist = dist[pos_match, :]

        # Get global index of the points
        points = points_select[pos_match]

        # If a candidate point is within a threshold to a source point, ignore it
        ignore = torch.where(dist < cl_cfg['ignore_distance'])
        ignore = torch.stack(ignore)

        neg_mask = torch.ones_like(dist)
        neg_mask[ignore[0, :], ignore[1, :]] = 0

        # If a points is beyond a threshold to a source point, treat it as a negative point
        neg_mask_double = torch.cat([neg_mask, neg_mask], dim=1)

        # Get the location of source points, (3, num_pos_points)
        select_points = foregrond_points[points, :].transpose(0, 1).type(torch.LongTensor)

        # q, k are of positive pairs, (num_pos_points, feat_channel)
        q_s_feat = s_feat[0, :, select_points[0, :], select_points[1, :], select_points[2, :]].transpose(0, 1)
        k_t_feat = t_feat[0, :, select_points[0, :], select_points[1, :], select_points[2, :]].transpose(0, 1)

        # Get foreground feature from source and target volumes
        s_foregrond_feat = s_feat[0, :, mask[0, 0, :, :, :] == 1]  # (channel, num_foreground_points)
        t_foregrond_feat = t_feat[0, :, mask[0, 0, :, :, :] == 1]  # (channel, num_foreground_points)

        # Inner product
        inner_view = torch.einsum("nc,nc->n", q_s_feat, k_t_feat).view(-1, 1)  # (num_pos_points, 1)

        # Calculate inner product within negative pairs, using q_s_feat as anchor
        neg_view_1 = torch.einsum("nc,ck->nk", q_s_feat, torch.cat((s_foregrond_feat, t_foregrond_feat), dim=1))
        neg_view_1 = neg_view_1 * neg_mask_double

        neg_candidate_view_index_1 = neg_view_1.topk(cl_cfg['pre_select_neg_number'], dim=1)[1]  # index

        neg_use_view_1 = torch.zeros((q_s_feat.shape[0], cl_cfg['after_select_neg_number']), device=neg_view_1.device)

        for i in range(q_s_feat.shape[0]):
            use_index = neg_candidate_view_index_1[i, torch.randperm(neg_candidate_view_index_1[i, :].shape[0]) \
                [:cl_cfg['after_select_neg_number']]]
            neg_use_view_1[i, :] = neg_view_1[i, use_index]

        logits_view_1 = torch.cat([inner_view, neg_use_view_1], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate inner product within negative pairs, using k_t_feat as anchor
        neg_view_2 = torch.einsum("nc,ck->nk", k_t_feat, torch.cat((s_foregrond_feat, t_foregrond_feat), dim=1))
        neg_view_2 = neg_view_2 * neg_mask_double

        neg_candidate_view_index_2 = neg_view_2.topk(cl_cfg['pre_select_neg_number'], dim=1)[1]  # index

        neg_use_view_2 = torch.zeros((k_t_feat.shape[0], cl_cfg['after_select_neg_number']), device=neg_view_2.device)

        for i in range(k_t_feat.shape[0]):
            use_index = neg_candidate_view_index_2[i, torch.randperm(neg_candidate_view_index_2[i, :].shape[0]) \
                [:cl_cfg['after_select_neg_number']]]
            neg_use_view_2[i, :] = neg_view_2[i, use_index]

        logits_view_2 = torch.cat([inner_view, neg_use_view_2], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate contrastive loss via CrossEntropy
        logits = torch.cat([logits_view_1, logits_view_2], dim=0)
        logits = logits / cl_cfg['fine_temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        contrastive_loss = self.criterion(logits, labels)
        return contrastive_loss

    def semantic_contrastive_loss(self, feat, mask):
        """
        Attention: Need to implement on full brain

        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        mask: (1, num_classes, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        mesh = meshgrid3d(mask[0, 0].shape).to(s_feat.device)  # (H, W, L, 3)

        # Get the foregound points (index) through mask
        foregrond_points = mesh[mask[0].sum(0) == 1, :]  # (num_points_in_mask, 3)

        # Select positive points, 'points_select' stores the index of selected points
        points_select = torch.randperm(foregrond_points.shape[0], device=s_feat.device)[
                        :cl_cfg["pre_select_pos_number"]]  # (num_pre_select_points)

        # Record the positive points' location
        select_points = foregrond_points[points_select, :].type(torch.LongTensor)  # (num_pre_select_points, 3)

        # Get the corresponding label of positive points
        continuous_seg = torch.argmax(mask, dim=1).squeeze()  # (H, W, L)
        positive_labels = continuous_seg[select_points[:, 0],
                                         select_points[:, 1],
                                         select_points[:, 2]]

        # Get the anchor feature of each class
        anchor_feat = []
        for idx in range(mask.shape[1]):
            points_of_class = mask[0, idx] == 1
            mean_vector = (s_feat[0, :, points_of_class].mean(-1) +
                           t_feat[0, :, points_of_class].mean(-1)) / 2
            # If the current brain doesn't contain {idx}th label, its mean_vector is nan
            anchor_feat.append(mean_vector)
        anchor_feat = torch.stack(anchor_feat)  # (num_classes, feat_channel)
        anchor_feat_mask = ~torch.isnan(anchor_feat.sum(-1))

        # q, k are of positive pairs, (num_pos_points, feat_channel)
        q_s_feat_1 = s_feat[0, :, select_points[:, 0], select_points[:, 1], select_points[:, 2]].transpose(0, 1)
        k_t_feat = anchor_feat[positive_labels]

        # Inner product
        inner_view_1 = torch.einsum("nc,nc->n", q_s_feat_1, k_t_feat).view(-1, 1)  # (num_pos_points, 1)

        neg_product_1 = torch.einsum("nc,kc->nk", q_s_feat_1,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_1 = torch.cat([inner_view_1, neg_product_1], dim=1)  # (num_pos_points, num_neg_points + 1)

        q_s_feat_2 = t_feat[0, :, select_points[:, 0], select_points[:, 1], select_points[:, 2]].transpose(0, 1)

        # Inner product
        inner_view_2 = torch.einsum("nc,nc->n", q_s_feat_2, k_t_feat).view(-1, 1)  # (num_pos_points, 1)

        neg_product_2 = torch.einsum("nc,kc->nk", q_s_feat_2,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_2 = torch.cat([inner_view_2, neg_product_2], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate contrastive loss via CrossEntropy
        logits = torch.cat([logits_view_1, logits_view_2], dim=0)  # (num_pos_points * 2, num_neg_points + 1)
        logits = logits / cl_cfg['fine_temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        contrastive_loss = self.criterion(logits, labels)
        return contrastive_loss

    def loss(self, feature_map, mask=None, ignore_label=None):
        """
        feature_map: (bsz, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        mask: (bsz, 1, H, W, L), can derive from segmentation map (bsz, nuw_class, H, W, L)
        """
        bsz, _, H, W, L = feature_map.shape
        if mask is None:
            mask = torch.ones([bsz, 1, H, W, L], dtype=torch.int, device=feature_map.device)
        elif mask.shape[1] > 1:
            seg = torch.clone(mask) if ignore_label is None else torch.clone(mask[:, ignore_label])
            # (bsz, nuw_class, H, W, L)
            mask = seg.sum(1, keepdim=True)  # (bsz, 1, H, W, L)

        if self.scale < 1:
            new_H, new_W, new_L = [int(old_size * self.scale) for old_size in feature_map.shape[2:]]
            h_start = np.random.randint(0, int(H - new_H), size=1).item()
            w_start = np.random.randint(0, int(W - new_W), size=1).item()
            l_start = np.random.randint(0, int(L - new_L), size=1).item()
            feature_map = feature_map[:, :,
                          h_start: (h_start + new_H),
                          w_start: (w_start + new_W),
                          l_start: (l_start + new_L)]
            mask = mask[:, :,
                   h_start: (h_start + new_H),
                   w_start: (w_start + new_W),
                   l_start: (l_start + new_L)]
            seg = seg[:, :,
                  h_start: (h_start + new_H),
                  w_start: (w_start + new_W),
                  l_start: (l_start + new_L)]

        cl_loss = torch.tensor(0., device=feature_map.device)
        for idx in range(bsz):
            batch_feat = feature_map[idx: idx + 1]
            batch_mask = mask[idx: idx + 1]
            batch_seg = seg[idx: idx + 1]
            # cl_loss += self.single_contrastive_loss(feat=batch_feat, mask=batch_mask)
            cl_loss += self.semantic_contrastive_loss(feat=batch_feat, mask=batch_seg)

        return cl_loss / bsz


class ContrastiveSem:
    def __init__(self, scale=1., norm=True):
        """
        scale: If you need to calculate CL loss within cropped volumes, set scale < 1.
        norm: Set True and it will normalize input feature
        """
        super().__init__()
        self.scale = scale
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def semantic_contrastive_loss(self, feat, seg1, seg2):
        """
        Select positive points randomly
        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        seg1: (1, num_classes, H, W, L)
        seg2: (1, num_classes, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        mesh = meshgrid3d(s_feat.shape[2:]).to(s_feat.device)  # (H, W, L, 3)

        # Get the foregound points (index) through mask
        foregrond_points_1 = mesh[seg1[0].sum(0) == 1, :]  # (num_points_in_mask, 3)
        foregrond_points_2 = mesh[seg2[0].sum(0) == 1, :]  # (num_points_in_mask, 3)

        # Select positive points, 'points_select' stores the index of selected points
        points_select_1 = torch.randperm(foregrond_points_1.shape[0], device=s_feat.device)[
                          :cl_cfg["pre_select_pos_number"]]  # (num_pre_select_points)
        points_select_2 = torch.randperm(foregrond_points_2.shape[0], device=s_feat.device)[
                          :cl_cfg["pre_select_pos_number"]]  # (num_pre_select_points)

        # Record the positive points' location
        select_points_1 = foregrond_points_1[points_select_1, :].type(torch.LongTensor)  # (num_pre_select_points, 3)
        select_points_2 = foregrond_points_2[points_select_2, :].type(torch.LongTensor)  # (num_pre_select_points, 3)

        # Get the corresponding label of positive points
        continuous_seg_1 = torch.argmax(seg1, dim=1).squeeze()  # (H, W, L)
        positive_labels_1 = continuous_seg_1[select_points_1[:, 0],
                                             select_points_1[:, 1],
                                             select_points_1[:, 2]]
        continuous_seg_2 = torch.argmax(seg2, dim=1).squeeze()  # (H, W, L)
        positive_labels_2 = continuous_seg_2[select_points_2[:, 0],
                                             select_points_2[:, 1],
                                             select_points_2[:, 2]]

        # Get the anchor feature of each class
        anchor_feat = []
        for idx in range(seg1.shape[1]):
            points_of_class_1 = seg1[0, idx] == 1
            s_mean = s_feat[0, :, points_of_class_1].mean(-1)
            s_mean_isnan = torch.isnan(s_mean).sum() > 0

            points_of_class_2 = seg2[0, idx] == 1
            t_mean = t_feat[0, :, points_of_class_2].mean(-1)
            t_mean_isnan = torch.isnan(t_mean).sum() > 0

            if s_mean_isnan and t_mean_isnan:
                mean_vector = s_mean
            else:
                mean_vector = (s_mean * ~s_mean_isnan + t_mean * ~t_mean_isnan) / \
                              ((~s_mean_isnan).float() + (~t_mean_isnan).float())
            # If the current brain doesn't contain {idx}th label, its mean_vector is nan
            anchor_feat.append(mean_vector)
        anchor_feat = torch.stack(anchor_feat)  # (num_classes, feat_channel)
        anchor_feat = F.normalize(anchor_feat, dim=1)
        anchor_feat_mask = ~torch.isnan(anchor_feat.sum(-1))

        # q, k are of positive pairs, (num_pos_points, feat_channel)
        q_s_feat_1 = s_feat[0, :, select_points_1[:, 0], select_points_1[:, 1],
                     select_points_1[:, 2]].transpose(0, 1)
        k_t_feat_1 = anchor_feat[positive_labels_1]

        # Inner product
        inner_view_1 = torch.einsum("nc,nc->n", q_s_feat_1, k_t_feat_1).view(-1, 1)  # (num_pos_points, 1)

        neg_product_1 = torch.einsum("nc,kc->nk", q_s_feat_1,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_1 = torch.cat([inner_view_1, neg_product_1], dim=1)  # (num_pos_points, num_neg_points + 1)

        q_s_feat_2 = t_feat[0, :, select_points_2[:, 0], select_points_2[:, 1],
                     select_points_2[:, 2]].transpose(0, 1)
        k_t_feat_2 = anchor_feat[positive_labels_2]

        # Inner product
        inner_view_2 = torch.einsum("nc,nc->n", q_s_feat_2, k_t_feat_2).view(-1, 1)  # (num_pos_points, 1)

        neg_product_2 = torch.einsum("nc,kc->nk", q_s_feat_2,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_2 = torch.cat([inner_view_2, neg_product_2], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate contrastive loss via CrossEntropy
        logits = torch.cat([logits_view_1, logits_view_2], dim=0)  # (num_pos_points * 2, num_neg_points + 1)
        logits = logits / cl_cfg['fine_temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        contrastive_loss = self.criterion(logits, labels)
        return contrastive_loss

    def semantic_contrastive_loss_v2(self, feat, seg1, seg2):
        """
        Select positive points based on ratio
        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        seg1: (1, num_classes, H, W, L)
        seg2: (1, num_classes, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        mesh = meshgrid3d(s_feat.shape[2:]).to(s_feat.device)  # (H, W, L, 3)

        # Get the anchor feature of each class
        anchor_feat = []
        vol_per_label_1 = torch.zeros([seg1.shape[1]], dtype=torch.int, device=seg1.device)
        vol_per_label_2 = torch.zeros([seg1.shape[1]], dtype=torch.int, device=seg1.device)
        for idx in range(seg1.shape[1]):
            points_of_class_1 = seg1[0, idx] == 1
            vol_per_label_1[idx] = points_of_class_1.sum()

            points_of_class_2 = seg2[0, idx] == 1
            vol_per_label_2[idx] = points_of_class_2.sum()

            s_mean = s_feat[0, :, points_of_class_1].mean(-1)
            s_mean_isnan = torch.isnan(s_mean).sum() > 0

            t_mean = t_feat[0, :, points_of_class_2].mean(-1)
            t_mean_isnan = torch.isnan(t_mean).sum() > 0

            if s_mean_isnan and t_mean_isnan:
                mean_vector = s_mean
            else:
                mean_vector = (s_mean * ~s_mean_isnan + t_mean * ~t_mean_isnan) / \
                              ((~s_mean_isnan).float() + (~t_mean_isnan).float())
            # If the current brain doesn't contain {idx}th label, its mean_vector is nan
            anchor_feat.append(mean_vector)
        anchor_feat = torch.stack(anchor_feat)  # (num_classes, feat_channel)
        anchor_feat = F.normalize(anchor_feat, dim=1)
        anchor_feat_mask = ~torch.isnan(anchor_feat.sum(-1))

        vol_ratio_1 = vol_per_label_1 / vol_per_label_1.sum()
        num_per_label_1 = (vol_ratio_1 * cl_cfg["pre_select_pos_number"]).int()
        num_per_label_1[1] += cl_cfg["pre_select_pos_number"] - num_per_label_1.sum()
        idx_per_label_1 = torch.tensor([0], dtype=torch.int, device=seg1.device)
        idx_per_label_1 = torch.cat((idx_per_label_1, num_per_label_1.cumsum(0)), dim=0)

        vol_ratio_2 = vol_per_label_2 / vol_per_label_2.sum()
        num_per_label_2 = (vol_ratio_2 * cl_cfg["pre_select_pos_number"]).int()
        num_per_label_2[1] += cl_cfg["pre_select_pos_number"] - num_per_label_2.sum()
        idx_per_label_2 = torch.tensor([0], dtype=torch.int, device=seg1.device)
        idx_per_label_2 = torch.cat((idx_per_label_2, num_per_label_2.cumsum(0)), dim=0)

        # Sample positive points
        select_points_1 = torch.zeros((cl_cfg["pre_select_pos_number"], 3)).type(torch.LongTensor)
        select_points_2 = torch.zeros((cl_cfg["pre_select_pos_number"], 3)).type(torch.LongTensor)
        for idx in range(seg1.shape[1]):
            # Get the foregound points (index) through mask of each label
            foregrond_points_1 = mesh[seg1[0, idx] == 1, :]  # (num_points_in_mask, 3)
            foregrond_points_2 = mesh[seg2[0, idx] == 1, :]  # (num_points_in_mask, 3)

            # Select positive points, 'points_select' stores the index of selected points
            points_select_1 = torch.randperm(foregrond_points_1.shape[0], device=s_feat.device)[
                              :num_per_label_1[idx].item()]  # (num_pre_select_points)
            points_select_2 = torch.randperm(foregrond_points_2.shape[0], device=s_feat.device)[
                              :num_per_label_2[idx].item()]  # (num_pre_select_points)

            # Record the positive points' location
            select_points_1_tmp = foregrond_points_1[points_select_1, :].type(
                torch.LongTensor)  # (num_pre_select_points, 3)
            select_points_2_tmp = foregrond_points_2[points_select_2, :].type(
                torch.LongTensor)  # (num_pre_select_points, 3)

            select_points_1[idx_per_label_1[idx].item(): idx_per_label_1[idx + 1].item()] = select_points_1_tmp
            select_points_2[idx_per_label_2[idx].item(): idx_per_label_2[idx + 1].item()] = select_points_2_tmp

        # Get the corresponding label of positive points
        continuous_seg_1 = torch.argmax(seg1, dim=1).squeeze()  # (H, W, L)
        positive_labels_1 = continuous_seg_1[select_points_1[:, 0],
                                             select_points_1[:, 1],
                                             select_points_1[:, 2]]
        continuous_seg_2 = torch.argmax(seg2, dim=1).squeeze()  # (H, W, L)
        positive_labels_2 = continuous_seg_2[select_points_2[:, 0],
                                             select_points_2[:, 1],
                                             select_points_2[:, 2]]
        # q, k are of positive pairs, (num_pos_points, feat_channel)
        q_s_feat_1 = s_feat[0, :, select_points_1[:, 0], select_points_1[:, 1],
                     select_points_1[:, 2]].transpose(0, 1)
        k_t_feat_1 = anchor_feat[positive_labels_1]

        # Inner product
        inner_view_1 = torch.einsum("nc,nc->n", q_s_feat_1, k_t_feat_1).view(-1, 1)  # (num_pos_points, 1)

        neg_product_1 = torch.einsum("nc,kc->nk", q_s_feat_1,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_1 = torch.cat([inner_view_1, neg_product_1], dim=1)  # (num_pos_points, num_neg_points + 1)

        q_s_feat_2 = t_feat[0, :, select_points_2[:, 0], select_points_2[:, 1],
                     select_points_2[:, 2]].transpose(0, 1)
        k_t_feat_2 = anchor_feat[positive_labels_2]

        # Inner product
        inner_view_2 = torch.einsum("nc,nc->n", q_s_feat_2, k_t_feat_2).view(-1, 1)  # (num_pos_points, 1)

        neg_product_2 = torch.einsum("nc,kc->nk", q_s_feat_2,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_2 = torch.cat([inner_view_2, neg_product_2], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate contrastive loss via CrossEntropy
        logits = torch.cat([logits_view_1, logits_view_2], dim=0)  # (num_pos_points * 2, num_neg_points + 1)
        logits = logits / cl_cfg['fine_temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        contrastive_loss = self.criterion(logits, labels)
        return contrastive_loss

    def semantic_contrastive_loss_v3(self, feat, seg1, seg2, dice):
        """
        Select positive points based on dice
        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        seg1: (1, num_classes, H, W, L)
        seg2: (1, num_classes, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        mesh = meshgrid3d(s_feat.shape[2:]).to(s_feat.device)  # (H, W, L, 3)

        # When selecting positive points, we don't need to calculate gradient
        with torch.no_grad():
            # Get the anchor feature of each class
            anchor_feat = []
            vol_per_label_min = torch.zeros([seg1.shape[1]], dtype=torch.int, device=seg1.device)
            for idx in range(seg1.shape[1]):
                points_of_class_1 = seg1[0, idx] == 1
                points_of_class_2 = seg2[0, idx] == 1
                vol_per_label_min[idx] = torch.min(points_of_class_1.sum(),
                                                   points_of_class_2.sum())

                s_mean = s_feat[0, :, points_of_class_1].mean(-1)
                s_mean_isnan = torch.isnan(s_mean).sum() > 0

                t_mean = t_feat[0, :, points_of_class_2].mean(-1)
                t_mean_isnan = torch.isnan(t_mean).sum() > 0

                if s_mean_isnan and t_mean_isnan:
                    mean_vector = s_mean
                else:
                    mean_vector = (s_mean * ~s_mean_isnan + t_mean * ~t_mean_isnan) / \
                                  ((~s_mean_isnan).float() + (~t_mean_isnan).float())
                # If the current brain doesn't contain {idx}th label, its mean_vector is nan
                anchor_feat.append(mean_vector)
            anchor_feat = torch.stack(anchor_feat)  # (num_classes, feat_channel)
            anchor_feat = F.normalize(anchor_feat, dim=1)
            anchor_feat_mask = ~torch.isnan(anchor_feat.sum(-1))
            # Select positive points by dice
            dice_ratio = torch.nn.functional.softmax(1 / (dice * 5), dim=-1)
            num_per_label = (dice_ratio * cl_cfg["pre_select_pos_number"]).int()
            num_per_label[num_per_label > vol_per_label_min] = vol_per_label_min[num_per_label > vol_per_label_min]
            if dice[1] < dice[0]:
                num_per_label[1] += cl_cfg["pre_select_pos_number"] - num_per_label.sum()
            else:
                num_per_label[0] += cl_cfg["pre_select_pos_number"] - num_per_label.sum()
            idx_per_label = torch.tensor([0], dtype=torch.int, device=seg1.device)
            idx_per_label = torch.cat((idx_per_label, num_per_label.cumsum(0)), dim=0)

            # Sample positive points
            select_points_1 = torch.zeros((cl_cfg["pre_select_pos_number"], 3)).type(torch.LongTensor)
            select_points_2 = torch.zeros((cl_cfg["pre_select_pos_number"], 3)).type(torch.LongTensor)
            for idx in range(seg1.shape[1]):
                # Get the foregound points (index) through mask of each label
                foregrond_points_1 = mesh[seg1[0, idx] == 1, :]  # (num_points_in_mask, 3)
                foregrond_points_2 = mesh[seg2[0, idx] == 1, :]  # (num_points_in_mask, 3)

                # Select positive points, 'points_select' stores the index of selected points
                points_select_1 = torch.randperm(foregrond_points_1.shape[0], device=s_feat.device)[
                                  :num_per_label[idx].item()]  # (num_pre_select_points)
                points_select_2 = torch.randperm(foregrond_points_2.shape[0], device=s_feat.device)[
                                  :num_per_label[idx].item()]  # (num_pre_select_points)

                # Record the positive points' location
                select_points_1_tmp = foregrond_points_1[points_select_1, :].type(
                    torch.LongTensor)  # (num_pre_select_points, 3)
                select_points_2_tmp = foregrond_points_2[points_select_2, :].type(
                    torch.LongTensor)  # (num_pre_select_points, 3)

                select_points_1[idx_per_label[idx].item(): idx_per_label[idx + 1].item()] = select_points_1_tmp
                select_points_2[idx_per_label[idx].item(): idx_per_label[idx + 1].item()] = select_points_2_tmp

            # Get the corresponding label of positive points
            continuous_seg_1 = torch.argmax(seg1, dim=1).squeeze()  # (H, W, L)
            positive_labels_1 = continuous_seg_1[select_points_1[:, 0],
                                                 select_points_1[:, 1],
                                                 select_points_1[:, 2]]
            continuous_seg_2 = torch.argmax(seg2, dim=1).squeeze()  # (H, W, L)
            positive_labels_2 = continuous_seg_2[select_points_2[:, 0],
                                                 select_points_2[:, 1],
                                                 select_points_2[:, 2]]

        # q, k are of positive pairs, (num_pos_points, feat_channel)
        q_s_feat_1 = s_feat[0, :, select_points_1[:, 0], select_points_1[:, 1],
                     select_points_1[:, 2]].transpose(0, 1)
        k_t_feat_1 = anchor_feat[positive_labels_1]

        # Inner product
        inner_view_1 = torch.einsum("nc,nc->n", q_s_feat_1, k_t_feat_1).view(-1, 1)  # (num_pos_points, 1)

        neg_product_1 = torch.einsum("nc,kc->nk", q_s_feat_1,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_1 = torch.cat([inner_view_1, neg_product_1], dim=1)  # (num_pos_points, num_neg_points + 1)

        q_s_feat_2 = t_feat[0, :, select_points_2[:, 0], select_points_2[:, 1],
                     select_points_2[:, 2]].transpose(0, 1)
        k_t_feat_2 = anchor_feat[positive_labels_2]

        # Inner product
        inner_view_2 = torch.einsum("nc,nc->n", q_s_feat_2, k_t_feat_2).view(-1, 1)  # (num_pos_points, 1)

        neg_product_2 = torch.einsum("nc,kc->nk", q_s_feat_2,
                                     anchor_feat[anchor_feat_mask])  # (num_pos_points, num_classes)
        logits_view_2 = torch.cat([inner_view_2, neg_product_2], dim=1)  # (num_pos_points, num_neg_points + 1)

        # Calculate contrastive loss via CrossEntropy
        logits = torch.cat([logits_view_1, logits_view_2], dim=0)  # (num_pos_points * 2, num_neg_points + 1)
        logits = logits / cl_cfg['fine_temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        contrastive_loss = self.criterion(logits, labels)
        return contrastive_loss

    def prototype_supcon(self, feat, seg1, seg2):
        """
        feature_map: (1, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        seg1: (1, num_classes, H, W, L)
        seg2: (1, num_classes, H, W, L)
        """
        channel = int(feat.shape[1] / 2)
        s_feat = feat[:, :channel]  # (1, C, H, W, L)
        t_feat = feat[:, channel:]  # (1, C, H, W, L)
        if self.norm:
            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

        contrastive_loss = []
        for idx in range(seg1.shape[1]):
            # Get prototypical vector
            points_of_class_1 = seg1[0, idx] == 1
            points_of_class_2 = seg2[0, idx] == 1
            s_mean = s_feat[0, :, points_of_class_1].mean(-1)
            t_mean = t_feat[0, :, points_of_class_2].mean(-1)
            mean_vector = F.normalize(((s_mean + t_mean) / 2), dim=0)[None, :]

            # Get positive points
            s_pos_feat = s_feat[0, :, points_of_class_1]  # (feat_channel, num_pos_points_1)
            t_pos_feat = t_feat[0, :, points_of_class_2]  # (feat_channel, num_pos_points_2)
            pos_feat = torch.cat([s_pos_feat, t_pos_feat], dim=1)
            inner_view = torch.matmul(mean_vector, pos_feat).transpose(0, 1)  # (num_pos_points, 1)
            if inner_view.shape[0] > cl_cfg['select_pos_num']:
                select_index = torch.randperm(inner_view.shape[0], device=s_feat.device)[
                               :cl_cfg['select_pos_num']]  # (num_select_points)
                inner_view = inner_view[select_index]

            # Get negative points
            neg_map_s = ~points_of_class_1 * seg1[0].sum(0).bool()
            s_neg_feat = s_feat[0, :, neg_map_s]  # (feat_channel, num_neg_points_1)
            neg_map_t = ~points_of_class_2 * seg2[0].sum(0).bool()
            t_neg_feat = t_feat[0, :, neg_map_t]  # (feat_channel, num_neg_points_2)
            neg_feat = torch.cat([s_neg_feat, t_neg_feat], dim=1)
            neg_view = torch.matmul(mean_vector, neg_feat)  # (1, num_neg_points)
            if neg_view.shape[1] > cl_cfg['select_neg_num']:
                select_index = torch.randperm(neg_view.shape[1], device=s_feat.device)[
                               :cl_cfg['select_neg_num']]  # (num_select_points)
                neg_view = neg_view[:, select_index]
            # Calculate contrastive loss via CrossEntropy
            logits = torch.cat([inner_view, neg_view.repeat(inner_view.shape[0], 1)],
                               dim=1)
            logits = logits / cl_cfg['fine_temperature']
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
            cl_per_label = self.criterion(logits, labels)
            contrastive_loss.append(cl_per_label)
        contrastive_loss = torch.stack(contrastive_loss)
        return contrastive_loss.mean()

    def loss(self, feature_map, seg_src, seg_tgt, ignore_label=None, each_dice=None):
        """
        feature_map: (bsz, C*2, H, W, L), the first 'C' is source's channel and the other is target's.
        seg_src: (bsz, num_class, H, W, L)
        seg_tgt: (bsz, num_class, H, W, L)
        """
        bsz, _, H, W, L = feature_map.shape
        seg_1 = torch.clone(seg_src)
        seg_2 = torch.clone(seg_tgt)

        if seg_src.shape[2] != feature_map.shape[2]:
            scale = feature_map.shape[2] / seg_src.shape[2]
            seg_1 = F.interpolate(seg_1, scale_factor=scale)
            seg_2 = F.interpolate(seg_2, scale_factor=scale)
            # seg_1 = F.interpolate(seg_1, scale_factor=scale, mode='trilinear')
            # seg_1 = F.one_hot(torch.argmax(seg_1, dim=1)).permute(0, 4, 1, 2, 3)
            # seg_2 = F.interpolate(seg_2, scale_factor=scale, mode='trilinear')
            # seg_2 = F.one_hot(torch.argmax(seg_2, dim=1)).permute(0, 4, 1, 2, 3)

        if ignore_label is not None:
            seg_1 = torch.clone(seg_1[:, ignore_label])
            seg_2 = torch.clone(seg_2[:, ignore_label])
        """
        if self.scale < 1:
            new_H, new_W, new_L = [int(old_size * self.scale) for old_size in feature_map.shape[2:]]
            h_start = np.random.randint(0, int(H - new_H), size=1).item()
            w_start = np.random.randint(0, int(W - new_W), size=1).item()
            l_start = np.random.randint(0, int(L - new_L), size=1).item()
            feature_map = feature_map[:, :,
                          h_start: (h_start + new_H),
                          w_start: (w_start + new_W),
                          l_start: (l_start + new_L)]
            seg_1 = seg_1[:, :,
                    h_start: (h_start + new_H),
                    w_start: (w_start + new_W),
                    l_start: (l_start + new_L)]
            seg_2 = seg_2[:, :,
                    h_start: (h_start + new_H),
                    w_start: (w_start + new_W),
                    l_start: (l_start + new_L)]
        """
        cl_loss = torch.tensor(0., device=feature_map.device)
        for idx in range(bsz):
            batch_feat = feature_map[idx: idx + 1]
            batch_seg_1 = seg_1[idx: idx + 1]
            batch_seg_2 = seg_2[idx: idx + 1]
            if each_dice is not None:
                dice_batch = each_dice[idx]
                cl_loss += self.semantic_contrastive_loss_v3(feat=batch_feat, seg1=batch_seg_1,
                                                             seg2=batch_seg_2, dice=dice_batch)
            else:
                cl_loss += self.semantic_contrastive_loss(feat=batch_feat, seg1=batch_seg_1, seg2=batch_seg_2)
                # cl_loss += self.semantic_contrastive_loss_v2(feat=batch_feat, seg1=batch_seg_1, seg2=batch_seg_2)
                # cl_loss += self.prototype_supcon(feat=batch_feat, seg1=batch_seg_1, seg2=batch_seg_2)

        return cl_loss / bsz
