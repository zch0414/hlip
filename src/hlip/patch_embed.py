import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224, 224), patch_size=(16, 16, 16), in_chans=1, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        assert len(list(img_size)) == 3, 'Specify the input size at every dimension'
        assert len(list(patch_size)) == 3, 'Specify the patch size at every dimension'

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=kwargs["bias"])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(-1, C, D, H, W)
        x = self.proj(x)
        _, _, D, H, W = x.shape
        
        # BN * C' * D' * H' * W' -> B * N * D' * H' * W' * C'
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, D, H, W, -1) 
        x = self.norm(x)
        return x