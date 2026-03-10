import torch
import torch.nn as nn
import torch.nn.functional as F
from prorandconv_orig import ProRandConvNet
from torchvision import models



# --- 1. SE-Block (通道注意力) ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 保证压缩后至少有1个通道
        reduced_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- 2. 投影头 (通道对齐 MLP) ---
class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. DoubleConv (支持空洞卷积 & 动态 Norm) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, norm_type='bn'):
        super().__init__()
        p = dilation
        
        # 动态选择 BN 或 IN
        if norm_type == 'in':
            NormLayer = lambda c: nn.InstanceNorm2d(c, affine=True)
        else:
            NormLayer = nn.BatchNorm2d

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=p, dilation=dilation),
            NormLayer(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=p, dilation=dilation),
            NormLayer(out_channels),
            nn.SiLU(inplace=True)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)
        return x

# --- 4. 主模型: MS-Fusion SE-U-Net ---
class MSFusionUNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=1, norm_type='bn', dilation=2):
        super().__init__()
        
        # 初始化 ProRandConvNet
        self.cls_net = models.resnet18(pretrained=True)
        self.cls_net = self.cls_net.cuda()
        self.prorandconv = ProRandConvNet(size=512).cuda()
        # 输入的rgb图片尺寸是512x512，所以ProRandConvNet的size参数设置为512，若设置为32，则会导致输入尺寸不匹配的错误。



        # [Step 1] Stem: 极小目标保护 (无下采样)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # [Step 2] Encoder
        self.enc1 = DoubleConv(64, 64, norm_type=norm_type)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128, norm_type=norm_type)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256, norm_type=norm_type)
        self.pool3 = nn.MaxPool2d(2)

        # [Step 3] Middle: 空洞卷积扩大视野
        self.middle = DoubleConv(256, 512, dilation=dilation, norm_type=norm_type)

        # [Step 4] Decoder (使用 BN 恢复特征)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, norm_type='bn') 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, norm_type='bn') 
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, norm_type='bn')

        # [Step 5] 多尺度投影 (Side Outputs)
        unified_channels = 64
        self.side3 = ProjectionHead(256, unified_channels)
        self.side2 = ProjectionHead(128, unified_channels)
        self.side1 = ProjectionHead(64,  unified_channels)

        # [Step 6] 最终融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(unified_channels * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    def forward_once(self, x):
        # 1. Stem
        x_stem = self.stem(x)

        # 2. Encoder
        e1 = self.enc1(x_stem)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # 3. Middle
        mid = self.middle(p3)

        # 4. Decoder
        d3 = self.up3(mid)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear')
        d3 = torch.cat([e3, d3], dim=1)
        d3_out = self.dec3(d3)

        d2 = self.up2(d3_out)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear')
        d2 = torch.cat([e2, d2], dim=1)
        d2_out = self.dec2(d2)

        d1 = self.up1(d2_out)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear')
        d1 = torch.cat([e1, d1], dim=1)
        d1_out = self.dec1(d1)

        # 5. Fusion
        target_size = x.shape[2:]
        s3 = F.interpolate(self.side3(d3_out), size=target_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.side2(d2_out), size=target_size, mode='bilinear', align_corners=True)
        s1 = self.side1(d1_out)

        fusion = torch.cat([s1, s2, s3], dim=1)
        out = self.fusion_conv(fusion)

        return out

    def forward(self, x):
        rgb = x[:, :3, :, :]
        other = x[:, 3:, :, :]

        rgb_rand = self.prorandconv(rgb, self.cls_net)

        x = torch.cat([rgb, other], dim=1)
        x_a = torch.cat([rgb_rand, other], dim=1)

        out = self.forward_once(x)
        out_a = self.forward_once(x_a)

        return out, out_a, rgb, rgb_rand
