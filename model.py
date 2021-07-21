import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # 3x3 kernel, stride 1, padding same
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        return self.conv(x)

# 2D UNET MODEL
class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet2D, self).__init__()
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet - encoder
        for feature in features:                            # adds 4 convblocks
            self.encode.append(DoubleConvBlock(in_channels, feature))   
            in_channels = feature

        # Up part of UNet - decoder
        for feature in reversed(features):
            self.decode.append(nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=4, padding = 1, stride=2,        # upsample
                )
            )
            self.decode.append(DoubleConvBlock(feature*2, feature))     # adds 4 conv blocks using reversed features

        self.bottleneck = DoubleConvBlock(features[-1], features[-1]*2)           # bottleneck
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)                
    
    def forward(self, x):
        skip_connections = []                                   # skip connections skip some layer in the  network and feeds the output of one layer as the input to the next layers (instead of only one)
        for down in self.encode:                                # for every conv block
            x = down(x)
            skip_connections.append(x)                          # append result to skip_connections list
            x = self.pool(x)                                    # apply maxpooling after double conv
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]               # reverse the order of skip connections
 
        for idx in range(0, len(self.decode), 2):               # step of 2
            x = self.decode[idx](x)                             # conv transpose
            skip_connection = skip_connections[idx//2]          # step by 2
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decode[idx+1](concat_skip)
        return self.final_conv(x)