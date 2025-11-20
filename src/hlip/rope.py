import torch
import torch.nn as nn
import torch.nn.functional as F


def keep_indices_to_dhw_pos(keep_indices: torch.LongTensor, spatial_shape: tuple) -> torch.LongTensor:
    """
    Convert kept flat indices into (d, h, w) coordinates.

    Args:
        keep_indices: LongTensor of shape [B, N_keep], values in [0, d*h*w - 1]
        spatial_shape: tuple (d, h, w) giving the original 3D grid size

    Returns:
        pos_dhw: LongTensor of shape [B, N_keep, 3], where the last dim is (d, h, w)
    """
    assert keep_indices.dtype == torch.long, "keep_indices must be LongTensor"
    assert keep_indices.dim() == 2, "keep_indices must be [B, N_keep]"
    d, h, w = spatial_shape
    B, N_keep = keep_indices.shape
    total = d * h * w
    if torch.any(keep_indices < 0) or torch.any(keep_indices >= total):
        raise ValueError(f"keep_indices out of range [0, {total-1}] for spatial shape {(d, h, w)}")

    # Flatten -> (d, h, w)
    dh = h * w
    d_idx = keep_indices // dh
    rem   = keep_indices % dh
    h_idx = rem // w
    w_idx = rem % w

    return d_idx, h_idx, w_idx


#https://github.com/facebookresearch/vjepa2/blob/c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb/src/models/utils/modules.py#L26
def rotate_queries_or_keys(x, pos):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    freq = torch.einsum("..., f -> ... f", pos.unsqueeze(1), omega)  # (..., N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (..., N, D/2)
    emb_cos = freq.cos()  # (..., N, D/2)
    # -- NOTE: This expansion has a subtle bug where frequencies are duplicated across the vector pair.
    # -- Fixing the bug would break compatibility with the pretrained model, but the fix can be applied by commenting
    # -- out the two lines below, and uncommenting the following two lines.
    # -- Thanks to @echosprint, original PR: https://github.com/facebookresearch/vjepa2/pull/15
    # emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)
    # emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)  # (..., N, D)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)  # (..., N, D)

    # --
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(
        dim=-1,
    )
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)