import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Vanilla U-Net for binary segmentation (1-channel in, 1-channel out)."""
    def __init__(self, base=64):
        super().__init__()
        self.down1 = DoubleConv(1, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        c3 = self.conv3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, c1], dim=1))

        logits = self.out(c1)  # NOTE: logits, no sigmoid here
        return logits
