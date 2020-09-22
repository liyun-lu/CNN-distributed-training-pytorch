import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class BasicBlock(nn.Module):  #定义残差模块
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if use_1x1conv:     #否使用额外的1×1卷积层来修改通道数
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels) #批处理正则化
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)

# 第一个参数layer_dims：resnet-34 [3, 4, 6, 3] 4个Res Block，每个包含2个Basic Block
# 第二个参数num_classes：我们的全连接输出，取决于输出有多少类。
class ResNet(nn.Module):
    def __init__(self, layer_dims, num_classes=1):
        super(ResNet, self).__init__()
        #预处理层。实现起来比较灵活可以加 MAXPool2D，也可以没有
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=3),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=3),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.build_block(64, 64, layer_dims[0], first_block=True)
        self.layer2 = self.build_block(64, 128, layer_dims[1])
        self.layer3 = self.build_block(128, 256, layer_dims[2])
        self.layer4 = self.build_block(256, 512, layer_dims[3])

        self.avgpool = nn.AvgPool2d(kernel_size=3)   #全局平局池化层
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def build_block(self, in_channels, out_channels, blocks, first_block=False):
        if first_block:
            assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(blocks):
            if i == 0 and not first_block:
                blk.append(BasicBlock(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*blk)

# 想要建立不同的resnet，在这里更改参数就行。ResNet-34是4个Res Block，第1个包含3个Basic Block,,第2为4，第3为6，第4为3
def resnet34():
    net = ResNet([3, 4, 6, 3])
    return net

# net = ResNet([3, 4, 6, 3])
# summary(net, (3, 330, 330))   #print shape



