

# Models for ViT Cycle GAN


# Design by HanLin


import warnings
warnings.filterwarnings('ignore')


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat
from timm.models.layers import to_2tuple, trunc_normal_


# Discriminator test 0 START =======


# In order to save resources, first try CNN model
class Discriminator(nn.Module):
    def __init__(self,input_channel):
        super(Discriminator,self).__init__()

        model = []

        model.extend([
            nn.Conv2d(input_channel,64,3,stride=2),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(64,128,3,stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(128,256,3,stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        model.extend([
            nn.Conv2d(256,512,3,stride=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        ])

        self.model = nn.Sequential(*model)

        self.adamaxpool = nn.AdaptiveMaxPool2d(1)

        self.linear = nn.Linear(512,1)

    def forward(self,x):
        x = self.model(x)
        x = self.adamaxpool(x)

        x = x.view(x.size(0),-1)

        x = self.linear(x)

        return x


# Discriminator test 0 END =======


# Generator test 0 START =======



class SE_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input X : B N C
        # X : B C N
        x = torch.transpose(x, 1, 2)
        # X : B C N
        b, c, n = x.size()
        # V : B C
        y = self.avg_pool(x).view(b, c)
        # V : B C 1
        y = self.fc(y).view(b, c, 1)
        # X : B C N
        x = x * y.expand_as(x)
        # X : B N C
        x = torch.transpose(x, 1, 2)

        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1,bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)

        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        v = self.to_v(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x):
        B_, N, C = x.shape

        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(x).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]

        return q, k, v


def window_partition(x,win_size):
    B,H,W,C = x.shape
    x = x.view(B,H//win_size,win_size,W//win_size,win_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,win_size,win_size,C)
    return windows

def window_reverse(windows,win_size,H,W):
    B = int(windows.shape[0]/(H*W/win_size/win_size))
    x = windows.view(B,H//win_size,W//win_size,win_size,win_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x


class Downsample_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        B, L, C = x.shape

        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C

        return out


class Upsample_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out


class WindowAttention(nn.Module):
    def __init__(self,dim,win_size,num_heads,token_projection='linear',qkv_bias=True,attn_drop=0.,proj_drop=0., se_layer=False):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SE_block(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
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


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),act_layer())
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)

        return x


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, window_size, window_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, window_size*window_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, window_size, window_size, 1
            shift_mask_windows = shift_mask_windows.view(-1,
                                                         self.win_size * self.win_size)  # nW, window_size*window_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, window_size*window_size, window_size*window_size
            attn_mask = attn_mask or shift_attn_mask
            attn_mask = attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        del attn_mask
        return x

class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',token_mlp='ffn',se_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for blk in self.blocks:

            x = blk(x,mask)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)

        return x


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C

        return x


class Generator(nn.Module):
    def __init__(self,
                 img_size=256,
                 in_chans=3,
                 embed_dim=16,
                 depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 token_projection='linear',
                 token_mlp='leff',
                 se_layer=False,
                 dowsample=Downsample_block,
                 upsample=Upsample_block
                 ):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size


        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,act_layer=nn.LeakyReLU)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      norm_layer=norm_layer,
                                      token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        y=x+y

        return y

# Generator test 0 END =======








