import numpy as np
from timm.layers import resample_abs_pos_embed as resample_2d_posemb

import torch
import torch.nn.functional as F


def resample_1d_posemb(posemb, num_samples, is_train=True):
    """resample study position embedding.
    """
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    _max = posemb.shape[1]

    # interpolate
    if _max < num_samples:
        assert not is_train
        posemb = F.interpolate(posemb.permute(0, 2, 1), size=num_samples, mode='linear').permute(0, 2, 1)
        return posemb.to(orig_dtype)
    
    # sample
    if num_samples <= _max:
        if is_train:
            perm = torch.randperm(_max)[:num_samples]
        else:
            perm = torch.arange(num_samples)
        posemb = posemb[:, perm, :]
        return posemb.to(orig_dtype)


def resample_3d_posemb(posemb, new_size, old_size):
    # new_size and old_size should be provided with the same shape: [d, h, w]
    # d: through-place dimension, h and w: in-place dimension.
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    if new_size == old_size:
        return posemb.to(orig_dtype)

    # interpolate
    if old_size[0] != new_size[0] or old_size[1] != new_size[1] or old_size[2] != new_size[2]:
        posemb = F.interpolate(posemb.permute(0, 4, 1, 2, 3), size=(new_size[0], new_size[1], new_size[2]), mode='trilinear').permute(0, 2, 3, 4, 1)

    return posemb.to(orig_dtype)


def study_pos_embed(max_num_scans, grid_size, embed_dim, pretrained_posemb=None): 
    """
    pretrained_posemb should be a 2D position embedding without prefix_posemb
    Return:
        spatial_posemb: A tensor of shape [1, d, h, w, embed_dim]
        sequential_posemb: A tensor of shape [1, max_num_scans, embed_dim]
    """
    if pretrained_posemb is not None: 
        # build mri position embedding from pretrained_posemb
        pretrained_posemb = resample_2d_posemb(
            pretrained_posemb,
            new_size=(grid_size[1], grid_size[2]),
            num_prefix_tokens=0 # enforce 0
        )
        pretrained_posemb = pretrained_posemb.reshape(1, grid_size[1], grid_size[2], embed_dim)
        slice_posemb = get_1d_sincos_pos_embed(
            embed_dim=embed_dim,
            sequence_len=grid_size[0],
            cls_token=False
        ) # [n, embed_dim]
        slice_posemb = torch.from_numpy(slice_posemb).float()
        spatial_posemb = slice_posemb[None, :, None, None, :] + pretrained_posemb[:, None, :, :, :]
    else: 
        # build mri position embedding from scratch
        d_posemb = get_1d_sincos_pos_embed(
            embed_dim=embed_dim,
            sequence_len=grid_size[0],
            cls_token=False
        ) # [n, embed_dim]
        d_posemb = torch.from_numpy(d_posemb).float()

        hw_posemb = get_2d_sincos_pos_embed(
            embed_dim=embed_dim, 
            grid_sizes=(grid_size[1], grid_size[2]),
            cls_token=False,
            flatten=False,
        ) # [h, w, embed_dim]
        hw_posemb = torch.from_numpy(hw_posemb).float()

        spatial_posemb = d_posemb[None, :, None, None, :] + hw_posemb[None, None, :, :, :]

    if max_num_scans > 1:
        sequential_posemb = get_1d_sincos_pos_embed(
            embed_dim=embed_dim,
            sequence_len=max_num_scans,
            cls_token=False,
        ) # [n, embed_dim]
        sequential_posemb = torch.from_numpy(sequential_posemb).float()[None, ...]
    else:
        sequential_posemb = None
            
    return spatial_posemb, sequential_posemb


def get_3d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False, flatten=True):
    """
    grid_sizes: sequence of the grid depth, height, and width
    return:
    pos_embed: [dot(grid_sizes), embed_dim] or [1+dot(grid_sizes), embed_dim] (w/ or w/o cls_token)
    """
    grid_d = np.arange(grid_sizes[0], dtype=np.float32)
    grid_h = np.arange(grid_sizes[1], dtype=np.float32)
    grid_w = np.arange(grid_sizes[2], dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_sizes[0], grid_sizes[1], grid_sizes[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if flatten:
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = pos_embed.reshape([grid_sizes[0], grid_sizes[1], grid_sizes[2], embed_dim])
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_sizes, cls_token=False, flatten=True):
    """
    grid_sizes: sequence of the grid height and width.
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_sizes[0], dtype=np.float32)
    grid_w = np.arange(grid_sizes[1], dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_sizes[0], grid_sizes[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if flatten:
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    else:
        pos_embed = pos_embed.reshape([grid_sizes[0], grid_sizes[1], embed_dim])
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, sequence_len, cls_token=False):
    """
    sequence_len: int of the sequence length
    return:
    pos_embed: [sequence_len, embed_dim] or [1+sequence_len, embed_dim] (w/ or w/o cls_token)
    """
    sequence = np.arange(sequence_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, sequence)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use one third of dimensions to encode grid_d
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D', D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D', D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D', D/3)

    emb = np.concatenate([emb_w, emb_h, emb_d], axis=1) # (H*W*D', D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb