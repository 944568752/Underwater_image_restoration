import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

                        # 进行宽为1的镜像填充
        conv_block = [  nn.ReflectionPad2d(1),
                        # 输入通道
                        # 输出通道
                        # 卷积尺寸
                        nn.Conv2d(in_features, in_features, 3),
                        # 进行归一化
                        # 层数
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        # 进行宽为1的镜像填充
                        nn.ReflectionPad2d(1),
                        # 输入通道
                        # 输出通道
                        # 卷积尺寸
                        nn.Conv2d(in_features, in_features, 3),
                        # 进行归一化
                        # 层数
                        nn.InstanceNorm2d(in_features)  ]
        # 容器，添加到容器中的网络模块将依次执行
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
                    # 进行宽为3的镜像填充
        model = [   nn.ReflectionPad2d(3),
                    # 输入通道
                    # 输出通道
                    # 卷积尺寸
                    nn.Conv2d(input_nc, 64, 7),
                    # 进行归一化
                    # 层数
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2

        for _ in range(2):
                                # 输入通道
                                # 输出通道
                                # 卷积尺寸
            model = model + [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                                # 进行归一化
                                # 层数
                                nn.InstanceNorm2d(out_features),
                                nn.ReLU(inplace=True) ]

            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model = model + [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2

        for _ in range(2):
                                # 逆卷积上采样
                                # 输入通道
                                # 输出通道
                                # 逆卷积尺寸
                                # 步长
                                # 填充
                                # 输出填充：使输出尺寸=步长*输入尺寸
            model = model + [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                                # 进行归一化
                                # 层数
                                nn.InstanceNorm2d(out_features),
                                nn.ReLU(inplace=True) ]

            in_features = out_features
            out_features = in_features//2

        # Output layer
                            # 进行宽为3的镜像填充
        model = model + [  nn.ReflectionPad2d(3),
                            # 输入通道
                            # 输出通道
                            # 卷积尺寸
                            nn.Conv2d(64, output_nc, 7),
                            nn.Tanh() ]
        # 容器，添加到容器中的网络模块将依次执行
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
                    # 输入通道
                    # 输出通道
                    # 卷积尺寸
                    # 步长
                    # 填充
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

                            # 输入通道
                            # 输出通道
                            # 卷积尺寸
                            # 步长
                            # 填充
        model = model + [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                            # 进行归一化
                            # 层数
                            nn.InstanceNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True) ]

                            # 输入通道
                            # 输出通道
                            # 卷积尺寸
                            # 步长
                            # 填充
        model = model + [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                            # 进行归一化
                            # 层数
                            nn.InstanceNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True) ]

                            # 输入通道
                            # 输出通道
                            # 卷积尺寸
                            # 填充
        model = model + [  nn.Conv2d(256, 512, 4, padding=1),
                            # 进行归一化
                            # 层数
                            nn.InstanceNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True) ]

                # FCN classification layer
                            # 输入通道
                            # 输出通道
                            # 卷积尺寸
                            # 填充
        model = model + [nn.Conv2d(512, 1, 4, padding=1)]
        # 容器，添加到容器中的网络模块将依次执行
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        # 平均池化
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)




