import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from flash_attn import flash_attn_varlen_kvpacked_func
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import pdb

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


# We implemented window area partition in this way for fast computation, here is the example of the amplification coefficient = 3
def window_area_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    d, h, w = D // window_size[0], H // window_size[1], W // window_size[2]
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, d, h, w, reduce(mul, window_size), C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    if B > 1:
        print('wrong')
        return -1
    x = windows[0]
    dim = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    a = F.pad(x, dim, "constant")

    b = torch.zeros(size=(d*h*w, 27, window_size[0], window_size[1], window_size[2], C)).cuda()
    b[:, 0] = a[:-2, :-2, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 1] = a[:-2, :-2, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 2] = a[:-2, :-2, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 3] = a[:-2, 1:-1, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 4] = a[:-2, 1:-1, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 5] = a[:-2, 1:-1, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 6] = a[:-2, 2:, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 7] = a[:-2, 2:, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 8] = a[:-2, 2:, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 9] = a[1:-1, :-2, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 10] = a[1:-1, :-2, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 11] = a[1:-1, :-2, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 12] = a[1:-1, 1:-1, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 13] = a[1:-1, 1:-1, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 14] = a[1:-1, 1:-1, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 15] = a[1:-1, 2:, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 16] = a[1:-1, 2:, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 17] = a[1:-1, 2:, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 18] = a[2:, :-2, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 19] = a[2:, :-2, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 21] = a[2:, :-2, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 21] = a[2:, 1:-1, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 22] = a[2:, 1:-1, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 23] = a[2:, 1:-1, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b[:, 24] = a[2:, 2:, :-2].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 25] = a[2:, 2:, 1:-1].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)
    b[:, 26] = a[2:, 2:, 2:].reshape(d*h*w, window_size[0], window_size[1], window_size[2], C)

    b = b.reshape(d*h*w, 3, 3, 3, window_size[0], window_size[1], window_size[2], C)
    b = b.permute(0, 1, 4, 2, 5, 3, 6, 7)
    b = b.reshape(-1, window_size[0] * 3, window_size[1] * 3, window_size[2] * 3, C)

    sb = reduce(mul, window_size)
    b = b.view(-1, sb * 27, C)
    return b


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
        D (int): Depth of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    return tuple(use_window_size)


class CrossWindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xa):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            xa: input features with shape of (num_windows*B, M, C)
        """
        B_, N, C = x.shape
        _, M, _ = xa.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(xa).reshape(B_, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        """
        # Flash Attention
        q = q.transpose(1, 2).reshape(-1, 1, 16)
        index_q = torch.arange(0, q.shape[0]+1, N, dtype=torch.int32).to(q.device)
        kv = kv.permute(1,3,0,2,4).reshape(-1, 2, 1, 16)
        index_kv = torch.arange(0, kv.shape[0]+1, M, dtype=torch.int32).to(q.device)
        x = flash_attn_varlen_kvpacked_func(q, kv, index_q, index_kv, N, M).reshape(B_, N, C)
        """
        # conventional Attention
        k, v = kv[0], kv[1]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 2, 2),
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossWindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, xa):
        B, D, H, W, C = x.shape
        window_size = get_window_size((D, H, W), self.window_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        xa = F.pad(xa, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # partition windows
        x_windows = window_partition(x, window_size)  # B*nW, Wd*Wh*Ww, C
        # x_area_windows = window_partition(xa, window_size)

        x_area_windows = window_area_partition(xa, window_size)
        # W-MSA/SW-MSA
        attn_windows = self.cross_attn(x_windows, x_area_windows)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, xa):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).

        """

        shortcut = x
        x = self.forward_part1(x, xa)
        x = shortcut + self.drop_path(x)

        x = x + self.forward_part2(x)

        return x
    

# Statement for "Full Transformer": stride=kernel size, convolution degenerates into learnable pooling or upsample
class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.down_conv = nn.Conv3d(dim, 2*dim, (2, 2, 2), stride=2, padding=0)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, D % 2))
            x = rearrange(x, 'b c d h w -> b d h w c')
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.down_conv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up_conv = nn.ConvTranspose3d(dim, dim//2, (2, 2, 2), stride=2, padding=0)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B,D,H,W,C
        """
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.up_conv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """ A basic down-sample Transformer encoding layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (7,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CrossTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, xa):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
            xa: Input feature a, tensor size (B, C, D, H, W).
        """
        # assert x.shape == xa.shape, "x xa must have same shape"
        # B, D, H, W, C = x.shape

        for blk in self.blocks:
            next_x = blk(x, xa)
            next_xa = blk(xa, x)
            x = next_a
            xa = next_xa

        if self.downsample is not None:
            x_down = self.downsample(x)
            xa_down = self.downsample(xa)
            return x, xa, x_down, xa_down
        return x, xa, x, xa


class BasicLayerUp(nn.Module):
    """ A basic up-sample Transformer encoder layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (7,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CrossTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, xa):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
            xa: Input feature a, tensor size (B, C, D, H, W).
        """
        # assert x.shape == xa.shape, "x xa must have same shape"
        # B, D, H, W, C = x.shape

        for blk in self.blocks:
            next_x = blk(x, xa)
            next_xa = blk(xa, x)
            x = next_a
            xa = next_xa

        if self.upsample is not None:
            x_up = self.upsample(x)
            xa_up = self.upsample(xa)
            return x, xa, x_up, xa_up
        return x, xa, x, xa


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class XMorpher(nn.Module):
    """
    structure: 4 encoding stages(BasicLayer) + 4 decoding stages(BasicLayerUp)
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=False,
                 patch_size=(4, 4, 4),
                 in_chans=1,
                 embed_dim=64,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.up_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in reversed(range(self.num_layers)):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** i_layer))

            up_layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchExpand if i_layer > 0 else None,
                use_checkpoint=use_checkpoint)
            self.up_layers.append(up_layer)
            self.concat_back_dim.append(concat_linear)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.norm2 = norm_layer(self.embed_dim*2)

        self.reverse_patch_embedding = nn.ConvTranspose3d(2*embed_dim, embed_dim // 2, (4, 4, 4), stride=4)

    def forward(self, moving, fixed):
        """Forward function."""

        moving = self.patch_embed(moving)
        fixed = self.patch_embed(fixed)

        moving = self.pos_drop(moving)
        fixed = self.pos_drop(fixed)

        moving = rearrange(moving, 'n c d h w -> n d h w c')
        fixed = rearrange(fixed, 'n c d h w -> n d h w c')

        features_moving = []
        features_fixed = []
        for layer in self.layers:
            moving_out, fixed_out, moving, fixed = layer(moving.contiguous(), fixed.contiguous())
            features_moving.append(moving_out)
            features_fixed.append(fixed_out)

        moving = self.norm(moving)
        fixed = self.norm(fixed)

        for inx, layer_up in enumerate(self.up_layers):
            if inx == 0:
                _, _, moving, fixed = layer_up(moving, fixed)
            else:
                if moving.shape != features_moving[3 - inx].shape:
                    moving = rearrange(moving, 'n d h w c -> n c d h w')
                    fixed = rearrange(fixed, 'n d h w c -> n c d h w')
                    B, D, W, H, C = features_moving[3 - inx].shape
                    moving = F.interpolate(moving, size=(D, W, H), mode='trilinear', align_corners=True)
                    fixed = F.interpolate(fixed, size=(D, W, H), mode='trilinear', align_corners=True)
                    moving = rearrange(moving, 'n c d h w -> n d h w c')
                    fixed = rearrange(fixed, 'n c d h w -> n d h w c')

                moving = torch.cat([moving, features_moving[3 - inx]], -1)
                fixed = torch.cat([fixed, features_fixed[3 - inx]], -1)
                moving = self.concat_back_dim[inx](moving)
                fixed = self.concat_back_dim[inx](fixed)
                _, _, moving, fixed = layer_up(moving, fixed)

        x = torch.cat([moving, fixed], -1)
        x = self.norm2(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        x = self.reverse_patch_embedding(x)  # reverse the patch embedding to transfer the final feature into image size

        return x


class Head(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.swin = XMorpher(window_size=(2, 2, 2), in_chans=n_channels, embed_dim=48)
        self.out_conv = nn.Conv3d(24, 3, 3, padding=1)
        self.stn = SpatialTransformer()
        self.rstn = Re_SpatialTransformer()

    def forward(self, moving, fixed, mov_label=None, fix_label=None):
        x = self.swin(fixed, moving)
        flow = self.out_conv(x)

        w_m_to_f = self.stn(moving, flow)
        w_f_to_m = self.rstn(fixed, flow)

        if mov_label is not None:
            w_label_m_to_f = self.stn(mov_label, flow, mode='nearest')
        else:
            w_label_m_to_f = None

        if fix_label is not None:
            w_label_f_to_m = self.rstn(fix_label, flow, mode='nearest')
        else:
            w_label_f_to_m = None

        return w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow
