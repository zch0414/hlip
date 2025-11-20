import os
import sys
sys.path.append(os.path.abspath('.'))

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models import register_model, build_model_with_cfg, checkpoint
from timm.models.vision_transformer import VisionTransformer, Block
from timm.layers import PatchDropout, Attention, maybe_add_mask

from .patch_embed import PatchEmbed3D
from .pos_embed import study_pos_embed, resample_1d_posemb, resample_3d_posemb
from .rope import rotate_queries_or_keys, keep_indices_to_dhw_pos


class HLIPAttentionRope(Attention):
    def __init__(self, **kwargs):
        self.head_dim = kwargs["dim"] // kwargs["num_heads"]


        super().__init__(**kwargs)

    def forward(self, x, npt, keep_indices, grid_size, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # ---
        # https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/modules.py
        d_idx, h_idx, w_idx = keep_indices_to_dhw_pos(keep_indices, grid_size)
        d_dim = int(2 * ((self.head_dim // 3) // 2))
        h_dim = int(2 * ((self.head_dim // 3) // 2))
        w_dim = int(2 * ((self.head_dim // 3) // 2))
        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[:, :, npt:, s : s + d_dim], pos=d_idx)
        kd = rotate_queries_or_keys(k[:, :, npt:, s : s + d_dim], pos=d_idx)
        s += d_dim
        # Rotate height
        qh = rotate_queries_or_keys(q[:, :, npt:, s : s + h_dim], pos=h_idx)
        kh = rotate_queries_or_keys(k[:, :, npt:, s : s + h_dim], pos=h_idx)
        s += h_dim
        # Rotate width
        qw = rotate_queries_or_keys(q[:, :, npt:, s : s + w_dim], pos=w_idx)
        kw = rotate_queries_or_keys(k[:, :, npt:, s : s + w_dim], pos=w_idx)
        s += w_dim
        # Combine rotated dimension
        q_pt = q[:, :, :npt, :]
        k_pt = k[:, :, :npt, :]
        if s < self.head_dim:
            qr = q[:, :, npt:, s:]
            kr = k[:, :, npt:, s:]
            q_vt = torch.cat([qd, qh, qw, qr], dim=-1)
            k_vt = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q_vt = torch.cat([qd, qh, qw], dim=-1)
            k_vt = torch.cat([kd, kh, kw], dim=-1)
        q = torch.cat([q_pt, q_vt], dim=2)
        k = torch.cat([k_pt, k_vt], dim=2)
        # ---

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HLIPBlockRope(Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # reset attn
        self.attn = HLIPAttentionRope(
            dim=kwargs["dim"],
            num_heads=kwargs["num_heads"],
            qkv_bias=kwargs["qkv_bias"],
            qk_norm=kwargs["qk_norm"],
            scale_norm=kwargs["scale_attn_norm"],
            proj_bias=kwargs["proj_bias"],
            attn_drop=kwargs["attn_drop"],
            proj_drop=kwargs["proj_drop"],
            norm_layer=kwargs["norm_layer"],
        )
    
    def forward(self, x, npt, keep_indices, grid_size, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), npt, keep_indices, grid_size, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class HLIPVisualEncoderRope(VisionTransformer):
    def __init__(self, **kwargs):
        max_num_scans = kwargs.pop('max_num_scans')
        self.slice_attn_indexes = kwargs.pop('slice_attn_indexes', ())
        self.scan_attn_indexes = kwargs.pop('scan_attn_indexes', ())
        self.study_attn_indexes = kwargs.pop('study_attn_indexes', ())
        self.global_pool_dinotxt = kwargs.pop('global_pool_dinotxt', False)
        super().__init__(**kwargs)
        
        # reset num_features
        if self.global_pool_dinotxt:
            self.num_features = 2 * self.embed_dim

        # reset patch_dropout to return keep_indices
        patch_drop_rate = kwargs.get('patch_drop_rate', 0.0)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True
            )
        else:
            self.patch_drop = nn.Identity()

        # reset pos_embed
        spatial_posemb, sequential_posemb = study_pos_embed(
            max_num_scans=max_num_scans,
            grid_size=self.patch_embed.grid_size,
            embed_dim=self.embed_dim,
            pretrained_posemb=None,
        )
        self.spatial_posemb = nn.Parameter(spatial_posemb)
        self.spatial_posemb.requires_grad = False
        if sequential_posemb is not None:
            self.sequential_posemb = nn.Parameter(sequential_posemb)
            self.sequential_posemb.requires_grad = False
        else:
            self.sequential_posemb = None

    def _pos_embed(self, x):
        # x: [bs, n, d, h, w, c]
        bs, n, d, h, w, _ = x.shape
        spatial_posemb = resample_3d_posemb(self.spatial_posemb, (d, h, w), self.patch_embed.grid_size)
        if self.sequential_posemb is not None:
            sequential_posemb = resample_1d_posemb(self.sequential_posemb, n, is_train = bs!=1)
            pos_embed = sequential_posemb[:, :, None, None, None, :] + spatial_posemb[:, None, :, :, :, :]
            pos_embed = pos_embed.expand(bs, -1, -1, -1, -1, -1)
        else:
            pos_embed = spatial_posemb[:, None, :, :, :, :].expand(bs, n, -1, -1, -1, -1)

        # start status for vit blocks
        if 0 in self.slice_attn_indexes:
            pos_embed = pos_embed.flatten(3, 4).flatten(0, 2) # [bs * n * d, h * w, c]
            x = x.flatten(3, 4).flatten(0, 2) # [bs * n * d, h * w, c]
        elif 0 in self.scan_attn_indexes:
            pos_embed = pos_embed.flatten(2, 4).flatten(0, 1) # [bs * n, d * h * w, c]
            x = x.flatten(2, 4).flatten(0, 1) # [bs * n, d * h * w, c]
        elif 0 in self.study_attn_indexes:
            pos_embed = pos_embed.flatten(1, 4) # [bs , n * d * h * w, c]
            x = x.flatten(1, 4) # [bs , n * d * h * w, c]

        x = self.pos_drop(x + pos_embed)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)

        return x
     
    def _slice2scan(self, x, num_slices, keep_indices):
        """
        Slice unpartition into the original scan.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_slices (int): number of slices in one scan.
            keep_indices (tensor): indices with [B * num_scans * num_slices, L].

        Returns:
            x: [B * num_scans, num_prefix_tokens + num_slices * L, C].
            keep_indices (tensor): indices with [B * num_scans, num_slices * L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x[:, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape

        prefix_tokens = prefix_tokens.view(BND//num_slices, num_slices, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND//num_slices, num_slices, L, C).view(BND//num_slices, num_slices * L, C)        

        x = torch.cat([prefix_tokens, src], dim=1)
        keep_indices = keep_indices.contiguous().view(BND//num_slices, num_slices, L).view(BND//num_slices, num_slices * L)
        return x, keep_indices
    
    def _slice2study(self, x, num_scans, num_slices, keep_indices):
        """
        Slices unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans * num_slices, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.
            num_slices (int): number of slices in on scan.
            keep_indices (tensor): indices with [B * num_scans * num_slices, L, C].

        Returns:
            x: [B, num_prefix_tokens + num_scans * num_slices * L, C].
            keep_indices: [B, num_scans * num_slices * L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BND, L, C = src.shape
        
        prefix_tokens = prefix_tokens.view(BND//(num_scans*num_slices), num_scans * num_slices, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BND//(num_scans*num_slices), num_scans * num_slices, L, C).view(BND//(num_scans*num_slices), num_scans * num_slices * L, C)
        
        x = torch.cat([prefix_tokens, src], dim=1)
        keep_indices = keep_indices.contiguous().view(BND//(num_scans*num_slices), num_scans * num_slices, L).view(BND//(num_scans*num_slices), num_scans * num_slices * L)
        return x, keep_indices

    def _scan2study(self, x, num_scans, keep_indices):
        """
        Scans unpartition into the original study.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + L, C].
            num_scans (int): number of scans in one study.
            keep_indices (tensor): indices with [B * num_scans, L]

        Returns:
            x: [B, num_prefix_tokens + num_scans * L, C].
            keep_indices: [B, num_scans * L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BN, L, C = src.shape
        
        prefix_tokens = prefix_tokens.view(BN//num_scans, num_scans, self.num_prefix_tokens, C).mean(dim=1)
        src = src.view(BN//num_scans, num_scans, L, C).view(BN//num_scans, num_scans * L, C)
        

        x = torch.cat([prefix_tokens, src], dim=1)
        keep_indices = keep_indices.contiguous().view(BN//num_scans, num_scans, L).view(BN//num_scans, num_scans * L)
        return x, keep_indices
    
    def _study2slice(self, x, num_scans, num_slices, keep_indices):
        """
        Study partition into non-overlapping slices.
        Args:
            x (tensor): input tokens with [B, num_prefix_tokens + num_scans * num_slices * L, C].
            num_scans (int): number of scans in one study.
            num_slices (int): number of slices in one scan.
            keep_indices (tensor): indices with [B, num_scans * num_slices * L, C].

        Returns:
            x: [B * num_scans * num_slices, num_prefix_tokens + L, C].
            keep_indices: [B * num_scans * num_slices, L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        B, NDL, C = src.shape

        prefix_tokens = prefix_tokens.view(B, 1, 1, self.num_prefix_tokens, C).expand(-1, num_scans, num_slices, -1, -1).contiguous()
        src = src.view(B, num_scans, num_slices, NDL//(num_scans*num_slices), C)

        x = torch.cat([prefix_tokens, src], dim=3)
        x = x.view(-1, self.num_prefix_tokens+NDL//(num_scans*num_slices), C)
        keep_indices = keep_indices.contiguous().view(B, num_scans, num_slices, NDL//(num_scans*num_slices)).view(B * num_scans * num_slices, NDL//(num_scans*num_slices))
        return x, keep_indices
    
    def _study2scan(self, x, num_scans, keep_indices):
        """
        Study partition into non-overlapping scans.
        Args:
            x (tensor): input tokens with [B, num_prefix_tokens + num_scans * L, C].
            num_scans (int): number of scans in one study.
            keep_indices (tensor): indices with [B, num_scans * L, C].

        Returns:
            x: [B * num_scans, num_prefix_tokens + L, C].
            keep_indices: [B * num_scans, L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        B, NL, C = src.shape

        prefix_tokens = prefix_tokens.view(B, 1, self.num_prefix_tokens, C).expand(-1, num_scans, -1, -1).contiguous()
        src = src.view(B, num_scans, NL//num_scans, C)
        
        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens+NL//num_scans, C)
        keep_indices = keep_indices.contiguous().view(B, num_scans, NL//num_scans).view(B * num_scans, NL//num_scans)
        return x, keep_indices

    def _scan2slice(self, x, num_slices, keep_indices):
        """
        Scan partition into non-overlapping slices.
        Args:
            x (tensor): input tokens with [B * num_scans, num_prefix_tokens + num_slices * L, C].
            num_slices (int): number of slices in one scan.
            keep_indices (tensor): indices with [B * num_scans, num_slices * L].

        Returns:
            x: [B * num_scans * num_slices, num_prefix_tokens + L, C].
            keep_indices: [B * num_scans * num_slices, L].
        """
        prefix_tokens, src = x[:, :self.num_prefix_tokens, :].contiguous(), x [:, self.num_prefix_tokens:, :].contiguous()
        BN, DL, C = src.shape

        prefix_tokens = prefix_tokens.view(BN, 1, self.num_prefix_tokens, C).expand(-1, num_slices, -1, -1).contiguous()
        src = src.view(BN, num_slices, DL//(num_slices), C)

        
        x = torch.cat([prefix_tokens, src], dim=2)
        x = x.view(-1, self.num_prefix_tokens+DL//num_slices, C)
        keep_indices = keep_indices.contiguous().view(BN, num_slices, DL//(num_slices)).view(BN * num_slices, DL//(num_slices))
        return x, keep_indices
        
    def forward_features(self, x):
        x = self.patch_embed(x) # [b, n, d, h, w, c]
        
        num_scans, num_slices, h, w = x.shape[1: -1]
        grid_size = (num_slices, h, w)
        x = self._pos_embed(x)

        if isinstance(self.patch_drop, nn.Identity):
            x = self.patch_drop(x)
            keep_indices = torch.arange(torch.prod(torch.tensor(grid_size)), device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        else:
            x, keep_indices = self.patch_drop(x)
            if keep_indices is None:
                keep_indices = torch.arange(torch.prod(torch.tensor(grid_size)), device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        
        x = self.norm_pre(x)

        for idx, blk in enumerate(self.blocks):
            if idx-1 in self.slice_attn_indexes and idx in self.study_attn_indexes:
                x, keep_indices = self._slice2study(x, num_scans, num_slices, keep_indices)
            elif idx-1 in self.slice_attn_indexes and idx in self.scan_attn_indexes:
                x, keep_indices = self._slice2scan(x, num_slices, keep_indices)
            elif idx-1 in self.scan_attn_indexes and idx in self.study_attn_indexes:
                x, keep_indices = self._scan2study(x, num_scans, keep_indices)
            elif idx-1 in self.study_attn_indexes and idx in self.slice_attn_indexes:
                x, keep_indices = self._study2slice(x, num_scans, num_slices, keep_indices)
            elif idx-1 in self.study_attn_indexes and idx in self.scan_attn_indexes:
                x, keep_indices = self._study2scan(x, num_scans, keep_indices)
            elif idx-1 in self.scan_attn_indexes and idx in self.slice_attn_indexes:
                x, keep_indices = self._scan2slice(x, num_slices, keep_indices)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, self.num_prefix_tokens, keep_indices, grid_size)
            else:
                x = blk(x, self.num_prefix_tokens, keep_indices, grid_size)

        if len(self.blocks) - 1 in self.scan_attn_indexes: 
            x, _ = self._scan2study(x, num_scans, keep_indices)
        if len(self.blocks) - 1 in self.slice_attn_indexes:
            x, _ = self._slice2study(x, num_slices, num_scans, keep_indices)

        return self.norm(x)

    # NOTE: DION.TXT's head
    def forward_head_dinotxt(self, x, pre_logits=False):
        prefix_tokens = x[:, 0:self.num_prefix_tokens].mean(dim=1)
        visual_tokens = x[:, self.num_prefix_tokens:].mean(dim=1)

        x = torch.cat([prefix_tokens, visual_tokens], dim=1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool_dinotxt:
            x = self.forward_head_dinotxt(x)
        else:
            x = self.forward_head(x)
        return x


def custom_checkpoint_filter_fn(state_dict, model, patch_size=(16, 16, 16)):
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)

    # determine whether the cls_token has corresponding pos_embed
    embed_len = state_dict['pos_embed'].shape[1]
    if torch.sqrt(torch.tensor(embed_len)) != torch.sqrt(torch.tensor(embed_len)).floor():
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        state_dict['pos_embed'] = state_dict['pos_embed'][:, 1:]

    for k, v in state_dict.items():
        if 'patch_embed' in k:
            if model.patch_embed.__class__ == PatchEmbed3D:
                if 'weight' in k:
                    if (v.shape[2], v.shape[3]) != (patch_size[1], patch_size[2]):
                        v = torch.nn.functional.interpolate(v, size=(patch_size[1], patch_size[2]), mode='bicubic')

                    # NOTE: average inflation initialization
                    v = v.sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1).div(patch_size[0])

                    # NOTE: central inflation initialization (hard code ablation)
                    # v = v.sum(dim=1, keepdim=True).unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1)
                    # v[:, :, :patch_size[0]//2, :, :] = 0
                    # if patch_size[0] > 2: 
                    #     v[:, :, patch_size[0]//2+1:, :, :] = 0
            else:
                continue
        if 'pos_embed' in k:
            spatial_posemb, _ = study_pos_embed(
                max_num_scans = 1,
                grid_size = model.patch_embed.grid_size,
                embed_dim = model.embed_dim,
                pretrained_posemb = v
            )
            out_dict['spatial_posemb'] = spatial_posemb
            continue
        out_dict[k] = v
    return out_dict


def custom_create_vision_transformer(variant, **kwargs):
    kwargs.pop('pretrained_cfg_overlay')
    return build_model_with_cfg(
        model_cls=HLIPVisualEncoderRope,
        variant=variant,
        pretrained_cfg_overlay=dict(first_conv=None),
        pretrained_strict=False,
        pretrained_filter_fn=partial(custom_checkpoint_filter_fn, patch_size=kwargs['patch_size']),
        **kwargs,
    )


@register_model
def ablate_rope_vit_base_multiscan_h2_dinotxt1568(pretrained=True, **kwargs):
    model_args = dict(
        max_num_scans=0, scan_attn_indexes=(0, 1, 3, 4, 6, 7, 9, 10), study_attn_indexes=(2, 5, 8, 11),
        global_pool_dinotxt=True,
        img_size=(48, 224, 224), patch_size=(6, 16, 16),
        in_chans=1, depth = 12, embed_dim=768, num_heads=12, num_classes=0, no_embed_class=True, pos_embed='none',
        embed_layer=PatchEmbed3D, block_fn=HLIPBlockRope,
    )
    model = custom_create_vision_transformer('vit_base_patch16_224.mae', pretrained=pretrained, **dict(model_args, **kwargs))
    return model