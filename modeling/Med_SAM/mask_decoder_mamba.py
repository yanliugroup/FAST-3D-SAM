import torch
import torch.nn as nn
import torch.nn.functional as F
#from mamba_ssm import Mamba

class MaskDecoderMamba(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, mlahead_channels=256, num_classes=2):
        super(MaskDecoder, self).__init__()
        dim = 256
        self.mamba_layers = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.cls = nn.Sequential(nn.Conv3d(dim, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 1))

    def forward(self, x):

        B, C, H, W, D = x.shape  # [2, 256, 16, 16, 16]
        x = x.reshape(B, C, -1) 
        x = x.permute(0, 2, 1) # [2, 4096, 256]

        x = self.mamba_layers(x)
        
        assert x.shape[-1] == 256  # [2, 4096, 256]
        x = x.permute(0, 2, 1) # [2, 256, 4096]
        x = x.reshape(B, C, H, W, D) # [2, 256, 16, 16, 16]

        x = F.interpolate(x, size = 64, mode='trilinear', align_corners=True)
        
        x = self.cls(x)
        
        x = F.interpolate(x, scale_factor = 2, mode='trilinear', align_corners=True)
        return x
    
    
    
class MaskDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, mlahead_channels=256, num_classes=2):
        super(MaskDecoder, self).__init__()
        
        channel_1 = mlahead_channels
        channel_2 = mlahead_channels // 2
        
        dim = 256
        # self.mamba_layers = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=dim, # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,    # Local convolution width
        #     expand=2,    # Block expansion factor
        # )
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(mlahead_channels, channel_1, 3, padding=1, groups=8),
            nn.InstanceNorm3d(channel_1),
            nn.ReLU()
            )
        
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(mlahead_channels, channel_1, 3, padding=1, stride=2, groups=8),
            nn.InstanceNorm3d(channel_1),
            nn.ReLU()
            )
        
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(mlahead_channels, channel_1, 4, stride=4, groups=8),
            nn.InstanceNorm3d(channel_1),
            nn.ReLU()
            )
        
        self.mlp1 = nn.Sequential(
            nn.Conv3d(channel_1, channel_2, 1),
            nn.InstanceNorm3d(channel_2),
            nn.ReLU()
            )

        self.cls = nn.Sequential(nn.Conv3d(channel_2 * 3, mlahead_channels, 1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 1))

    def forward(self, input, ret_feats=False):
        
        # B, C, H, W, D = input.shape  # [2, 256, 16, 16, 16]
        # input = input.reshape(B, C, -1) 
        # input = input.permute(0, 2, 1) # [2, 4096, 256]

        # #input = self.mamba_layers(input)
        
        # assert input.shape[-1] == 256  # [2, 4096, 256]
        # input = input.permute(0, 2, 1) # [2, 256, 4096]
        # input = input.reshape(B, C, H, W, D) # [2, 256, 16, 16, 16]
        
        
        # for inp in inputs:
        #     print("mask decoder", inp.shape)
        #print("input", input.shape)
        
        x0 = self.conv0(input)
        x1 = self.up_conv1(input)
        x2 = self.up_conv2(input)
        
        x0 = self.mlp1(x0)
        x1 = self.mlp1(x1)
        x2 = self.mlp1(x2)
        
        #print("x0", x0.shape)
        x0 = F.interpolate(x0, size = 64, mode='trilinear', align_corners=True)
        x1 = F.interpolate(x1, size = 64, mode='trilinear', align_corners=True)
        
        #torch.Size([2, 256, 16, 16, 16]) torch.Size([2, 256, 31, 31, 31]) torch.Size([2, 256, 64, 64, 64])
        #torch.Size([2, 256, 64, 64, 64]) torch.Size([2, 256, 64, 64, 64]) torch.Size([2, 256, 64, 64, 64])

        x = torch.cat([x0, x1, x2], dim=1)
        
        x = self.cls(x)
        
        x = F.interpolate(x, scale_factor = 2, mode='trilinear', align_corners=True)
        
        if ret_feats:
            return x, [x0, x1, x2]
        else:
            return x