from torch import nn
import torch

class SqueezeExcitation2D(nn.Module):
    
    def __init__(self, channels):
        super(SqueezeExcitation2D, self).__init__()
        
        self.channels = int(channels/2)
        
        self.se = nn.Sequential(
            nn.Conv2d(channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(self.channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid())
        
    def forward(self, x):
        
        y = self.se(x)
        out = x*y
        
        return out

class ResidualBlockTypeB(nn.Module):
    
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlockTypeB, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, outchannel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        
        self.shortcut = shortcut
        self.se = SqueezeExcitation2D(outchannel)
        self.outchannel = outchannel
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.shortcut is None:
            residual = x
        else:
            residual = self.shortcut(x)
        
        out += residual
        out = self.se(out)
    
        return out
    
class ResidualBlockTypeA(nn.Module):
    
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlockTypeA, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, outchannel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.conv3 = nn.Conv2d(outchannel, outchannel, 1, 2, bias=False)
        
        self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, 2),
                nn.BatchNorm2d(outchannel))
        self.se = SqueezeExcitation2D(outchannel)
        
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)
    
        residual = self.shortcut(x)
        
        out += residual
        out = self.se(out)
        
        return out

class ResNet34(nn.Module):
    
    def __init__(self):
        super(ResNet34, self).__init__()
        
        self.pre = nn.Sequential(
                nn.Conv2d(1, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        
        self.layer1 = self._make_layer( 128, 128, 3)
        
        self.residual_a1 = ResidualBlockTypeA(128, 128, 2)
        
        self.layer2 = self._make_layer( 128, 128, 3, stride=2)
        
        self.residual_a2_a = ResidualBlockTypeA(128, 256, 2)
        
        self.layer3 = self._make_layer( 256, 256, 5, stride=1)
        
        self.residual_a2_b = ResidualBlockTypeA(256, 256, 2)
        
        self.layer4 = self._make_layer( 256, 256, 2, stride=1)
        
        self.attention = nn.Sequential(
            nn.Conv1d(2560, 640, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Conv1d(640, 2560, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.fc = nn.Linear(5120, 256)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlockTypeB(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlockTypeB(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.pre(x)
        x = self.layer1(x)
        x = self.residual_a1(x)
        x = self.layer2(x)
        x = self.residual_a2_a(x)
        x = self.layer3(x)
        x = self.residual_a2_b(x)
        x = self.layer4(x)
        x = x.view(2560, -1)
        
        w = self.attention(x)

        mean = torch.sum(x*w, dim=2)
        sigma = torch.sqrt(torch.sum((x**2)*w, dim=2)-mean**2)

        x = torch.cat((mean, sigma), 1)
        
        return self.fc(x)