import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    # in_channels and out_channels in unet main paper are as follows:
    # in_channels = [n, 64, 128, 256, 512, 1024, 512, 256, 128, 64]
    # out_channels = [64, 128, 256, 512, 1024, 512, 256, 128, 64, m]
    def __init__(self, in_channels, out_channels, skip_connection=True):
        super().__init__()
        self.down1 = DownSampling(in_channels=in_channels[0], out_channels=out_channels[0])
        self.down2 = DownSampling(in_channels=in_channels[1], out_channels=out_channels[1])
        self.down3 = DownSampling(in_channels=in_channels[2], out_channels=out_channels[2])
        self.down4 = DownSampling(in_channels=in_channels[3], out_channels=out_channels[3])
        self.bottom = DownSampling(in_channels=in_channels[4], out_channels=out_channels[4], apply_pool=False)

        self.up1 = UpSampling(in_channels=in_channels[5], out_channels=out_channels[5], skip_connection=skip_connection)
        self.up2 = UpSampling(in_channels=in_channels[6], out_channels=out_channels[6], skip_connection=skip_connection)
        self.up3 = UpSampling(in_channels=in_channels[7], out_channels=out_channels[7], skip_connection=skip_connection)
        self.up4 = UpSampling(in_channels=in_channels[8], out_channels=out_channels[8], skip_connection=skip_connection)
        self.final_conv = nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels[-1], kernel_size=1)

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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
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
    def __init__(self, in_channels, out_channels, skip_connection):
        super().__init__()
        self.skip_connection = skip_connection
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
        if self.skip_connection:
            diff_y = skip_connection.size(2) - x.size(2)
            diff_x = skip_connection.size(3) - x.size(3)
            skip_connection = F.pad(
                skip_connection,
                [-diff_x // 2, -diff_x + diff_x // 2, -diff_y // 2, -diff_y + diff_y // 2]
            )
            x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)











