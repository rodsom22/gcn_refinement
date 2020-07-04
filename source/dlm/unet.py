import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np


class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, dropout_rate=0.5):
        super(UnetConv, self).__init__()
        self.drop = nn.Dropout2d(p=dropout_rate)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv1.weight.data.normal_(0, np.sqrt(2./(kernel_size*kernel_size*in_channels)))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2.weight.data.normal_(0, np.sqrt(2./(kernel_size*kernel_size*out_channels)))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)        

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        self.drop(y)
        y = functional.relu(y)
    
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        y = functional.relu(y)
        return y


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, dropout_rate=0.5):
        super(UnetDown, self).__init__()
        self.conv = UnetConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding=padding, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        ft = self.conv(x)
        y = self.pool(ft)
        return y, ft


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, dropout_rate=0.5):
        super(UnetUp, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = UnetConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding=padding, dropout_rate=dropout_rate)

    def forward(self, xdown, xup):
        y = self.upconv(xup)
        y = crop_and_concat_tensor(xdown, y)
        y = self.conv(y)            
        return y


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, kernel_size=3, padding=0, dropout_rate=0.5):
        super(UNet, self).__init__()
        self.down1 = UnetDown(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, padding=padding,
                              dropout_rate=dropout_rate)
        self.down2 = UnetDown(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding,
                              dropout_rate=dropout_rate)
        self.down3 = UnetDown(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding,
                              dropout_rate=dropout_rate)
        self.down4 = UnetDown(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=padding,
                              dropout_rate=dropout_rate)
        self.middleconv = UnetConv(in_channels=512, out_channels=1024, kernel_size=kernel_size, padding=padding,
                                   dropout_rate=dropout_rate)
        self.up1 = UnetUp(in_channels=1024, out_channels=512, kernel_size=kernel_size, padding=padding,
                          dropout_rate=dropout_rate)
        self.up2 = UnetUp(in_channels=512, out_channels=256, kernel_size=kernel_size, padding=padding,
                          dropout_rate=dropout_rate)
        self.up3 = UnetUp(in_channels=256, out_channels=128, kernel_size=kernel_size, padding=padding,
                          dropout_rate=dropout_rate)
        self.up4 = UnetUp(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=padding,
                          dropout_rate=dropout_rate)
        self.outconv = nn.Conv2d(in_channels=64, out_channels=out_classes, kernel_size=1)

    def forward(self, x):
        y, down1 = self.down1(x)
        y, down2 = self.down2(y)
        y, down3 = self.down3(y)
        y, down4 = self.down4(y)
        y = self.middleconv(y)
        y = self.up1(down4, y)
        y = self.up2(down3, y)
        y = self.up3(down2, y)
        ft = self.up4(down1, y)
        logits = self.outconv(ft)
        return logits, ft


def crop_and_concat_tensor(x1, x2):
    """
    Crops x1 to the x2's size and concatenate the tensors across the channels. Is assumed that x1 is bigger than x2.
    The tensors have shape [batch, channels, y, x]
    """
    x_off = (x1.size()[3] - x2.size()[3]) // 2
    y_off = (x1.size()[2] - x2.size()[2]) // 2
    xs = x2.size()[3]
    ys = x2.size()[2]
    x_crop = x1[:, :, y_off:y_off+ys, x_off:x_off+xs]
    x = torch.cat((x_crop, x2), dim=1)
    return x


def crop_label_to_size(x1, x2):
    """
    Crops x1 (labels shaped [batch, y, x]) to the x2's (logits shaped [batch, c, y, x]) size and
    concatenate the tensors across the channels. Is assumed that x1 is bigger than x2. The tensors have shape
    [batch, y, x]
    """
    x_off = (x1.size()[2] - x2.size()[3]) // 2
    y_off = (x1.size()[1] - x2.size()[2]) // 2
    xs = x2.size()[3]
    ys = x2.size()[2]
    x = x1[:, y_off:y_off+ys, x_off:x_off+xs]
    return x



