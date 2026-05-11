import logging

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from detectors import DETECTOR
from einops import rearrange
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.hub import load_state_dict_from_url

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='tall')
class TALLDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.model = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        model_kwargs = dict(num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                            mlp_ratio=config['mlp_ratio'], patch_size=config['patch_size'],
                            window_size=config['window_size'], depths=config['depths'],
                            num_heads=config['num_heads'], ape=config['ape'],
                            thumbnail_rows=config['thumbnail_rows'], drop_rate=config['drop_rate'],
                            drop_path_rate=config['drop_path_rate'], use_checkpoint=False, bottleneck=False,
                            duration=config['clip_size'])
        default_cfg = {
            'url': config['pretrained'],
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
            'crop_pct': .9, 'interpolation': 'bicubic',
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'patch_embed.proj', 'classifier': 'head', }
        backbone = SwinTransformer(img_size=config['resolution'], **model_kwargs)
        backbone.default_cfg = default_cfg
        load_pretrained(backbone, num_classes=config['num_classes'], in_chans=model_kwargs.get('in_chans', 3),
                        filter_fn=_conv_filter, img_size=config['resolution'], pretrained_window_size=7,
                        pretrained_model='')

        return backbone

    def build_loss(self, config):
                                   
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        bs, t, c, h, w = data_dict['image'].shape
        inputs = data_dict['image'].view(bs, t * c, h, w)
        return self.model(inputs)

    def classifier(self, features: torch.Tensor):
        pass

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].long()
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        pred = self.features(data_dict)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': prob}


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


def window_partition(x, window_size):
\
\
\
\
\
\
\
       
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
\
\
\
\
\
\
\
\
\
       
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size          
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

                                                            
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))                       
                                                                                
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))             
        coords_flatten = torch.flatten(coords, 1)            
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]                   
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()                   
        relative_coords[:, :, 0] += self.window_size[0] - 1                         
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
\
\
\
\
           
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                                       

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)                  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()                    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
                                                             
        flops = 0
                           
        flops += N * self.dim * 3 * self.dim
                                          
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
                         
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
                          
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, bottleneck=False, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        if min(self.input_resolution) <= self.window_size:
                                                                                        
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
                                                 
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))           
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)                                   
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward_attn(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

                      
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

                           
        x_windows = window_partition(shifted_x, self.window_size)                                     
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)                                    

                      
        attn_windows = self.attn(x_windows, mask=self.attn_mask)                                    

                       
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)             

                              
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        return x

    def forward_mlp(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_attn, x)
        else:
            x = self.forward_attn(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_mlp, x)
        else:
            x = x + self.forward_mlp(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
               
        flops += self.dim * H * W
                      
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
             
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
               
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
\
\
\
\
\
\
       

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
\
\
           
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]               
        x1 = x[:, 1::2, 0::2, :]               
        x2 = x[:, 0::2, 1::2, :]               
        x3 = x[:, 1::2, 1::2, :]               
        x = torch.cat([x0, x1, x2, x3], -1)                 
        x = x.view(B, -1, 4 * C)                 

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 bottleneck=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

                      
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 bottleneck=bottleneck if i == depth - 1 else False,
                                 use_checkpoint=use_checkpoint)
            for i in range(depth)])

                             
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = sum(blk.flops() for blk in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
\
\
\
\
\
\
\
\
       

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
                                        
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
                                                 
        assert H == self.img_size[0] and W == self.img_size[1],\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)             
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, duration=8, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, thumbnail_rows=1, bottleneck=False, **kwargs):
        super().__init__()

        self.duration = duration     
        self.num_classes = num_classes     
        self.num_layers = len(depths)                 
        self.embed_dim = embed_dim       
        self.ape = ape        
        self.patch_norm = patch_norm         
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio               
        self.thumbnail_rows = thumbnail_rows     

        self.img_size = img_size       
        self.window_size = window_size if isinstance(window_size, list) else [window_size for _ in depths]
                                                                 

        self.frame_padding = self.duration % thumbnail_rows     
        if self.frame_padding != 0:
            self.frame_padding = self.thumbnail_rows - self.frame_padding
            self.duration += self.frame_padding

                                                  
        thumbnail_dim = (thumbnail_rows, self.duration // thumbnail_rows)          
        thumbnail_size = (img_size * thumbnail_dim[0], img_size * thumbnail_dim[1])

        self.patch_embed = PatchEmbed(
            img_size=(img_size, img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches      
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution            

                                     
        if self.ape:        
            self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.duration, embed_dim))
            trunc_normal_(self.frame_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

                          
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]                               

                      
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=self.window_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               bottleneck=bottleneck)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        return {'absolute_pos_embed', 'frame_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def create_thumbnail(self, x):
                                    
        input_size = x.shape[-2:]
        if input_size != to_2tuple(self.img_size):
            x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.thumbnail_rows, c=3)
        return x

    def pad_frames(self, x):
        frame_num = self.duration - self.frame_padding
        x = x.view((-1, 3 * frame_num) + x.size()[2:])
        x_padding = torch.zeros((x.shape[0], 3 * self.frame_padding) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.duration, 'frame number %d not the same as adjusted input size %d' % (
            x.shape[1], 3 * self.duration)

        return x

                                                             
    def create_image_pos_embed(self):
        img_rows, img_cols = self.patches_resolution            
        _, _, T = self.frame_pos_embed.shape                 
        rows = img_rows // self.thumbnail_rows      
        cols = img_cols // (self.duration // self.thumbnail_rows)      
        img_pos_embed = torch.zeros(img_rows, img_cols, T).cuda()                   
        for i in range(self.duration):
            r_indx = (i // self.thumbnail_rows) * rows
            c_indx = (i % self.thumbnail_rows) * cols
            img_pos_embed[r_indx:r_indx + rows, c_indx:c_indx + cols] = self.frame_pos_embed[0, i]

        return img_pos_embed.reshape(-1, T)                  

    def forward_features(self, x):
        if self.frame_padding > 0:
            x = self.pad_frames(x)
        else:
            x = x.view((-1, 3 * self.duration) + x.size()[2:])

        x = self.create_thumbnail(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear')                    

        x = self.patch_embed(x)                     
        if self.ape:
            img_pos_embed = self.create_image_pos_embed()
            x = x + img_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)         
        x = self.avgpool(x.transpose(1, 2))         
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224, num_patches=196,
                    pretrained_window_size=7, pretrained_model="", strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    if len(pretrained_model) == 0:
                                                                                         
        state_dict = load_state_dict_from_url(cfg['url'], map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info(f'Converting first conv ({conv1_name}) pretrained weights from 3 to 1 channel')
        conv1_weight = state_dict[f"{conv1_name}.weight"]
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
                                               
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[f"{conv1_name}.weight"] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[f"{conv1_name}.weight"]
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning(f'Deleting first conv ({conv1_name}) from pretrained weights.')
            del state_dict[f"{conv1_name}.weight"]
            strict = False
        else:
            _logger.info(f'Repeating first conv ({conv1_name}) weights in channel dim.')
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[f"{conv1_name}.weight"] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
                                                                                                    
        classifier_weight = state_dict[f"{classifier_name}.weight"]
        state_dict[f"{classifier_name}.weight"] = classifier_weight[1:]
        classifier_bias = state_dict[f"{classifier_name}.bias"]
        state_dict[f"{classifier_name}.bias"] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:                                   
                                                                                                           
        del state_dict['model'][f"{classifier_name}.weight"]
        del state_dict['model'][f"{classifier_name}.bias"]
        strict = False
    '''
    ## Resizing the positional embeddings in case they don't match
    if img_size != cfg['input_size'][1]:
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
   '''

                                           
    window_size = (model.window_size)[0]
    print(pretrained_window_size, window_size)

    new_state_dict = state_dict['model'].copy()
    for key in state_dict['model']:
        if 'attn_mask' in key:
            del new_state_dict[key]

        if 'relative_position_index' in key:
            del new_state_dict[key]

                   
        if 'relative_position_bias_table' in key:
            pretrained_table = state_dict['model'][key]
            pretrained_table_size = int(math.sqrt(pretrained_table.shape[0]))
            table_size = int(math.sqrt(model.state_dict()[key].shape[0]))
            if pretrained_table_size != table_size:
                table = pretrained_table.permute(1, 0).view(1, -1, pretrained_table_size, pretrained_table_size)
                table = nn.functional.interpolate(table, size=table_size, mode='bilinear')
                table = table.view(-1, table_size ** 2).permute(1, 0)
                new_state_dict[key] = table

    for key in model.state_dict():
        if 'bottleneck_norm' in key:
            attn_key = key.replace('bottleneck_norm', 'norm1')
                                   
            new_state_dict[key] = new_state_dict[attn_key]

    print('loading weights....')
                          
    model.load_state_dict(new_state_dict, strict=False)


def _conv_filter(state_dict, patch_size=4):
                                                                                    
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
