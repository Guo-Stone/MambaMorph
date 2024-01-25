import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
import sys

sys.path.append(r"/home/guotao/code/voxelmorph-dev/voxelmorph/torch")
import layers
from modelio import LoadableModel, store_config_args
import node
from XMorpher import CrossTransformerBlock3D

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 norm=False,
                 conv_1_1=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            conv_1_1: If prev_nf != nf, use 1x1 conv. This function is used to compare with node.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf and conv_1_1:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                    convs.append(ConvBlock(ndims, nf, nf, norm=norm))
                else:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf and conv_1_1:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                    convs.append(ConvBlock(ndims, nf, nf, norm=norm))
                else:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            if prev_nf != nf and conv_1_1:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                self.remaining.append(ConvBlock(ndims, nf, nf, norm=norm))
            else:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class NodeUnet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 norm=False,
                 tol=1e-3,
                 adjoint=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                conv_layer = NodeConvBlock(ndims, nf, nf, norm=norm)
                convs.append(node.ODEBlock(conv_layer, tol=tol, adjoint=adjoint))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                if prev_nf != nf:
                    convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
                conv_layer = NodeConvBlock(ndims, nf, nf, norm=norm)
                convs.append(node.ODEBlock(conv_layer, tol=tol, adjoint=adjoint))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            if prev_nf != nf:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm, kernel=1, padding=0))
            conv_layer = NodeConvBlock(ndims, nf, nf, norm=norm)
            self.remaining.append(node.ODEBlock(conv_layer, tol=tol, adjoint=adjoint))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 neuralode=False,
                 tol=1e-3,
                 adjoint=True
                 ):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core unet model
        if neuralode:
            self.unet_model = NodeUnet(
                inshape,
                infeats=(src_feats + trg_feats),
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                tol=tol,
                adjoint=adjoint,
            )
        else:
            self.unet_model = Unet(
                inshape,
                infeats=(src_feats + trg_feats),
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                conv_1_1=False,
            )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, return_pos_flow=True, return_feature=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
            return_pos_flow: Return pos_flow or not
            return_feature: null
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        ret = {'moved_vol': y_source, 'preint_flow': preint_flow}  # Dict of return values
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if self.bidir:
            ret['moved_target'] = y_target
        return ret


class VxmFeat(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Version 2: Add feature extraction layer before concatenate two volumes
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_feat_extractor=None,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 unet_half_res=False,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure feature extraction model
        self.feature_extractor = Unet(
            inshape,
            infeats=1,
            nb_features=nb_feat_extractor,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1] // 2, 3))

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=nb_feat_extractor[-1][-1] * 2,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            conv_1_1=False,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, return_pos_flow=True,
                return_feature=False, return_warped_feat=False):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            return_pos_flow: Return posint flow.
            return_feature: Return feature.
            return_warped_feat: Return warped feature.
        """
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        # concatenate inputs and propagate unet
        x = torch.cat([source_feat, target_feat], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        ret = {'moved_vol': y_source, 'preint_flow': preint_flow}  # Dict of return values
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if return_warped_feat:
            warped_feature = self.transformer(source_feat, pos_flow)
            for head in self.read_out_head:
                warped_feature = head(warped_feature)
            ret['warped_feature'] = warped_feature
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(source_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        if self.bidir:
            ret['moved_target'] = y_target
        return ret


class VxmFeatFusion(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Version 2: Add feature extraction layer before concatenate two volumes
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_feat_extractor=None,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 unet_half_res=False,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure feature extraction model
        self.feature_extractor = Unet(
            inshape,
            infeats=1,
            nb_features=nb_feat_extractor,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1] // 2, 3))

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=nb_feat_extractor[-1][-1] * 2,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            conv_1_1=False,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

        # feature fusion block
        self.fusion = CrossTransformerBlock3D(nb_feat_extractor[-1][-1], num_heads=1)

    def forward(self, source, target, return_pos_flow=True,
                return_feature=False, return_warped_feat=False):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            return_pos_flow: Return posint flow.
            return_feature: Return feature.
            return_warped_feat: Return warped feature.
        """
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        source_down = F.interpolate(source_feat, scale_factor=0.5)
        target_down = F.interpolate(target_feat, scale_factor=0.5)
        source_attn = self.fusion(source_down.permute(0, 2, 3, 4, 1),
                                  target_down.permute(0, 2, 3, 4, 1))
        target_attn = self.fusion(source_down.permute(0, 2, 3, 4, 1),
                                  target_down.permute(0, 2, 3, 4, 1))
        source_attn = F.interpolate(source_attn.permute(0, 4, 1, 2, 3), scale_factor=2)
        target_attn = F.interpolate(target_attn.permute(0, 4, 1, 2, 3), scale_factor=2)
        source_feat = source_feat * source_attn
        target_feat = target_feat * target_attn

        # concatenate inputs and propagate unet
        x = torch.cat([source_feat, target_feat], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        ret = {'moved_vol': y_source, 'preint_flow': preint_flow}  # Dict of return values
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if return_warped_feat:
            warped_feature = self.transformer(source_feat, pos_flow)
            for head in self.read_out_head:
                warped_feature = head(warped_feature)
            ret['warped_feature'] = warped_feature
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(source_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        if self.bidir:
            ret['moved_target'] = y_target
        return ret


class VxmFeatDouble(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Version 2: Add feature extraction layer before concatenate two volumes
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_feat_extractor=None,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 unet_half_res=False,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure feature extraction model
        self.feature_extractor = Unet(
            inshape,
            infeats=1,
            nb_features=nb_feat_extractor,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        self.feature_extractor_2 = Unet(
            inshape,
            infeats=1,
            nb_features=nb_feat_extractor,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=1))
        self.read_out_head.append(ConvBlock(ndims, nb_feat_extractor[-1][-1] // 2, 3))

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=nb_feat_extractor[-1][-1] * 2,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            conv_1_1=False,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, return_pos_flow=True, return_feature=False):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
            return_both_flow: Return preint_flow and pos_flow.
        """
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor_2(target)

        # concatenate inputs and propagate unet
        x = torch.cat([source_feat, target_feat], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        ret = {'moved_vol': y_source, 'preint_flow': preint_flow}  # Dict of return values
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(source_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        if self.bidir:
            ret['moved_target'] = y_target
        return ret


class FeatureExtractor(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Version 2: Add feature extraction layer before concatenate two volumes
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_feat_extractor=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 unet_half_res=False,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure feature extraction model
        self.feature_extraction = Unet(
            inshape,
            infeats=1,
            nb_features=nb_feat_extractor,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

    def forward(self, source, target):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        """
        source_feat = self.feature_extraction(source)
        target_feat = self.feature_extraction(target)

        return torch.cat([source_feat, target_feat], dim=1)


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.zeros_(m.weight)
        # nn.init.normal_(m.weight, 0, 1e-5)

    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class VxmDDS(LoadableModel):
    """
    Differential data scorer for VoxelMorph.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 norm=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.norm = norm

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            norm=self.norm
        )

        # Norm
        if self.norm:
            BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.batch_norm = BatchNorm(self.unet_model.final_nf)

        # Linear layer
        feature_dimension = self.unet_model.final_nf

        for idx in inshape:
            feature_dimension *= idx
        self.linear = nn.Linear(feature_dimension, 1)

        # initialization
        torch.nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.unet_model.apply(weights_init)

    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        Return
            logits of image pair
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)  # (bsz, C, W, H, L)
        if self.norm:
            x = self.batch_norm(x)
        # get logits
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class VxmDDS2(LoadableModel):
    """
    Differential data scorer for VoxelMorph.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 norm=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.norm = norm
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            norm=self.norm
        )
        # Norm
        if self.norm:
            BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.batch_norm = BatchNorm(self.unet_model.final_nf)

        # Linear layer
        self.global_pool_avg = nn.functional.adaptive_avg_pool3d
        self.global_pool_max = nn.functional.adaptive_max_pool3d

        self.linear = nn.Linear(self.unet_model.final_nf, 1)

        # initialization
        torch.nn.init.xavier_normal_(self.linear.weight)
        # self.linear.weight = nn.Parameter(Normal(0, 1e-5).sample(self.linear.weight.shape))
        nn.init.zeros_(self.linear.bias)

        self.unet_model.apply(weights_init)

    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        Return
            logits of image pair
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)
        if self.norm:
            x = self.batch_norm(x)
        # get logits
        x = self.global_pool_max(x, tuple([int(img_size / 2) for img_size in tuple(x.shape[2:])]))
        x = self.global_pool_avg(x, (1, 1, 1))  # (bsz, C, 1, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel=3,
                 stride=1, padding=1, norm=False, act=True):
        super().__init__()
        self.norm = norm
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, padding)
        self.act = act
        if self.norm:
            BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.batch_norm = BatchNorm(out_channels)
        if self.act:
            self.activation = nn.LeakyReLU(0.2)
        torch.nn.init.kaiming_normal_(self.main.weight)
        torch.nn.init.constant_(self.main.bias, 0)

    def forward(self, x):
        out = self.main(x)
        if self.norm:
            out = self.batch_norm(out)
        if self.act:
            out = self.activation(out)
        return out


class NodeConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel=3, stride=1, norm=False):
        super().__init__()
        self.norm = norm
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, 1)
        if self.norm:
            BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
            self.batch_norm = BatchNorm(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        torch.nn.init.kaiming_normal_(self.main.weight)
        torch.nn.init.constant_(self.main.bias, 0)

    def forward(self, t, x):
        out = self.main(x)
        if self.norm:
            out = self.batch_norm(out)
        out = self.activation(out)
        return out
