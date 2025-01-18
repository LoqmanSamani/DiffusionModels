import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DownSampling(in_channels=in_channels, out_channels=64)
        self.down2 = DownSampling(in_channels=64, out_channels=128)
        self.down3 = DownSampling(in_channels=128, out_channels=256)
        self.down4 = DownSampling(in_channels=256, out_channels=512)
        self.bottom = DownSampling(in_channels=512, out_channels=1024, apply_pool=False)

        self.up1 = UpSampling(in_channels=1024, out_channels=512)
        self.up2 = UpSampling(in_channels=512, out_channels=256)
        self.up3 = UpSampling(in_channels=256, out_channels=128)
        self.up4 = UpSampling(in_channels=128, out_channels=64)

        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x, _ = self.bottom(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final_conv(x)






class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, apply_pool=True):
        super().__init__()
        self.apply_pool = apply_pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip_connection = x
        if self.apply_pool:
            x = self.pool(x)
        return x, skip_connection


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):

        x = self.up(x)
        diff_y = skip_connection.size(2) - x.size(2)
        diff_x = skip_connection.size(3) - x.size(3)
        skip_connection = F.pad(
            skip_connection,
            [-diff_x // 2, -diff_x + diff_x // 2, -diff_y // 2, -diff_y + diff_y // 2]
        )
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)











