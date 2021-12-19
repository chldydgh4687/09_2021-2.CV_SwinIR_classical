import torch.nn as nn
from model.STL import SwinTransformerLayer as STL
from model.TF_embed import PatchEmbed, PatchUnEmbed
import torch.utils.checkpoint as checkpoint

########################################
"""
    # RSTB (classical.ver)
    ---MultiSwinTransformerLayer(STLs)
    |----------STL
    |----------STL
    |----------STL
    |----------STL
    |----------STL
    |----------STL
    ----------conv
"""
########################################

class ResidualSwinTransformerBlocks(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection="1conv"):
        super(ResidualSwinTransformerBlocks,self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = MultiSwinTransformerLayer(dim=dim,
                                                        input_resolution=input_resolution,
                                                        depth=depth,
                                                        num_heads=num_heads,
                                                        window_size=window_size,
                                                        mlp_ratio=mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop, attn_drop=attn_drop,
                                                        drop_path=drop_path,
                                                        norm_layer=norm_layer,
                                                        downsample=downsample,
                                                        use_checkpoint=use_checkpoint)

        if resi_connection =="1conv":
            self.conv = nn.Conv2d(dim,dim,3,1,1)
        elif resi_connection =="3conv":
            self.conv = nn.Sequential(nn.Conv2d(dim,dim//4,3,1,1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim //4, dim //4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim//4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                      norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                                      norm_layer=None)
    def forward(self, x, x_size):
        #print(x.shape)
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x,x_size),x_size))) + x


# "BasicLayer" in referenced code
class MultiSwinTransformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            STL(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i%2 == 0) else window_size //2 ,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
        for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self,x,x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk,x,x_size)
            else:
                x = blk(x,x_size)

        if self.downsample is not None:
            x = self.downsample(x)

        return x




