import torch.nn as nn
from torch import nn
import torch
import torch.distributed as dist
import math

from .pytorch_borzoi_utils import *
from .pytorch_borzoi_transformer import *

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def lecun_normal_init(x):
    with torch.no_grad():
        fan_in = x.weight.size(1) * x.weight.size(2)
        gain = 1.0
        scale = gain / fan_in
        stddev = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(x.weight, std = stddev)

  
class ConvDna(nn.Module):
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels = 4,out_channels = 512, kernel_size = 15, padding="same")
        nn.init.zeros_(self.conv_layer.bias)
        lecun_normal_init(self.conv_layer)
        self.max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels=None, kernel_size=1,
                 conv_type="standard"):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels, padding = 'same', bias = False)
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            nn.init.zeros_(pointwise_conv.bias)
            lecun_normal_init(pointwise_conv)
            lecun_normal_init(depthwise_conv)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm =  nn.BatchNorm1d(in_channels,eps = 0.001)#
            self.activation =nn.GELU(approximate='tanh')
            self.conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding='same')
            nn.init.zeros_(self.conv_layer.bias)
            lecun_normal_init(self.conv_layer)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

    
class Borzoi(nn.Module):
    
    def __init__(self, checkpoint_path = None, 
                 enable_mouse_head = False,
                 flash_attention = False,
                 return_center_bins_only = False, 
                 rotary_emb_base=20000.0, 
                 rotary_emb_scale_base = None, 
                 transformer_dropout = 0.2, 
                 attn_dropout= 0.0, 
                 num_transformer_layers = 8,
                 num_transformer_heads = 8,
                 transformer_activation = 'relu', 
                 disable_heads = False):
        super(Borzoi, self).__init__()
        self.enable_mouse_head = enable_mouse_head
        self.return_center_bins_only = return_center_bins_only
        self.conv_dna = ConvDna()
        self._max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)
        self.res_tower = nn.Sequential(
            ConvBlock(in_channels = 512,out_channels = 608,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 608,out_channels = 736,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 736,out_channels = 896,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 896,out_channels = 1056,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 1056,out_channels = 1280,kernel_size = 5),
        )
        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(in_channels = 1280,out_channels = 1536,kernel_size = 5),
        )
        transformer = []
        for _ in range(num_transformer_layers):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(1536, eps = 0.001),
                    Attention(
                        1536,
                        heads = 8,
                        dim_key = 64,
                        dim_value = 192,
                        dropout = 0.05,
                        pos_dropout = 0.01,
                        num_rel_pos_features = 32
                    ) if not flash_attention else
                    FlashAttention(
                        dim=1536,
                        heads = num_transformer_heads,
                        dropout = attn_dropout,
                        pos_dropout = 0.,
                        rotary_emb_base=rotary_emb_base,
                        rotary_emb_scale_base = rotary_emb_scale_base,
                    ),
                    nn.Dropout(transformer_dropout))
                ),
                Residual(
                    nn.Sequential(
                        nn.LayerNorm(1536, eps = 0.001),
                        nn.Linear(1536, 1536 * 2),
                        nn.Dropout(transformer_dropout),
                        nn.ReLU() if transformer_activation == 'relu' else nn.GELU(),
                        nn.Linear(1536 * 2, 1536),
                        nn.Dropout(transformer_dropout)
                    ) 
                ) 
            )
        )
        self.horizontal_conv0,self.horizontal_conv1 = ConvBlock(in_channels = 1280, out_channels = 1536, kernel_size = 1),ConvBlock(in_channels = 1536, out_channels = 1536,kernel_size = 1)
        self.upsample = torch.nn.Upsample(scale_factor = 2)
        self.transformer = nn.Sequential(*transformer)
        for x in self.transformer: 
            nn.init.kaiming_normal_(x[1].fn[1].weight, nonlinearity = 'relu') 
            nn.init.kaiming_normal_(x[1].fn[4].weight, nonlinearity = 'relu') 
            nn.init.zeros_(x[1].fn[1].bias) 
            nn.init.zeros_(x[1].fn[4].bias) 
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels = 1536, out_channels = 1536,  kernel_size = 1),
            self.upsample,
        )
        self.separable1 = ConvBlock(in_channels = 1536, out_channels = 1536,  kernel_size = 3, conv_type = 'separable')
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels = 1536,out_channels = 1536,kernel_size = 1),
            self.upsample,
        )
        self.separable0 = ConvBlock(in_channels = 1536, out_channels = 1536,  kernel_size = 3, conv_type = 'separable')
        if self.return_center_bins_only:
            self.crop = TargetLengthCrop(6144)
        else:
            self.crop = TargetLengthCrop(16384-32)
        self.final_joined_convs = nn.Sequential(
            ConvBlock(in_channels = 1536, out_channels = 1920, kernel_size = 1),
            nn.Dropout(0.1),
            nn.GELU(approximate='tanh'),
        )
        self.human_head = nn.Conv1d(in_channels = 1920, out_channels = 7611, kernel_size = 1)
        nn.init.zeros_(self.human_head.bias)
        lecun_normal_init(self.human_head)
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(in_channels = 1920, out_channels = 2608, kernel_size = 1)
            nn.init.zeros_(self.mouse_head.bias)
            lecun_normal_init(self.mouse_head)
        self.final_softplus = nn.Softplus()
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))
        self.disable_heads = disable_heads

        
        
    def forward(self, x, is_human = True):
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)
        x = self._max_pool(x_unet1)
        x_unet1 = self.horizontal_conv1(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0)
        x = self.transformer(x.permute(0,2,1))
        x = x.permute(0,2,1)
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        x = self.crop(x.permute(0,2,1))
        x = self.final_joined_convs(x.permute(0,2,1))
        if not self.disable_heads:
            if self.compiled_head_softplus:
                if not is_human:
                    return self.mouse_head_softplus(x) + 0 * self.human_head_softplus(x).mean()
                else:
                    return self.human_head_softplus(x) + 0 * self.mouse_head_softplus(x).mean()
            if torch.is_autocast_enabled():
                with torch.cuda.amp.autocast(enabled=False):
                    if not is_human:
                        mouse_out = self.final_softplus(self.mouse_head(x.float())) + 0 * self.human_head(x.float()).sum()
                        return mouse_out
                    else:
                        human_out = self.final_softplus(self.human_head(x.float())) + 0 * self.mouse_head(x.float()).sum()
                        return human_out
            else:
                if not is_human:
                    mouse_out = self.final_softplus(self.mouse_head(x.float())) + 0 * self.human_head(x.float()).sum()
                    return mouse_out
                else:
                    human_out = self.final_softplus(self.human_head(x.float())) + 0 * self.mouse_head(x.float()).sum()
                    return human_out