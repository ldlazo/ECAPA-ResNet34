import math
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, bottleneck):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, x):
        y = self.se(x)
        output = x*y
        return output

class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, dilation=None, scale=8):

        super(BottleNeck, self).__init__()

        width = int(math.floor(out_channels/scale))
        
        self.conv1 = nn.Conv1d(in_channels, width*scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width*scale)
        
        self.index = scale -1 

        conv_list = [] 
        bn_list = [] 
        
        
        dil_fac = [1, 2, 3, 4, 5, 6, 2, 3]

        for i in range(self.index):
            pad = math.floor(kernel_size/2)*dil_fac[i]
            conv_list.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dil_fac[i], padding=pad))
            bn_list.append(nn.BatchNorm1d(width))

        self.conv_list = nn.ModuleList(conv_list)
        self.bn_list = nn.ModuleList(bn_list)

        self.conv2 = nn.Conv1d(width*scale, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.width = width
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        split_indexes = torch.split(out, self.width, 1)
        for i in range(self.index):
          if i==0:
            vec = split_indexes[i]
          else:
            vec = vec + split_indexes[i]
          vec = self.conv_list[i](vec)
          vec = self.relu(vec)
          vec = self.bn_list[i](vec)
          
          if i==0:
            out = vec
          else:
            out = torch.cat((out, vec), 1)

        out = torch.cat((out, split_indexes[self.index]),1)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out = self.se(out)

        out += residual

        return out 

class Res2Net(nn.Module):

    def __init__(self, block, in_channel, scale, out_dim, n_mfcc, **kwargs):

        super(Res2Net, self).__init__()
        self.scale = scale
        self.n_mfcc = n_mfcc

        self.conv = nn.Conv1d(self.n_mfcc, in_channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(in_channel)
        
        self.layer1 = block(in_channel, in_channel, kernel_size=3, dilation=2, scale=self.scale)
        self.layer2 = block(in_channel, in_channel, kernel_size=3, dilation=3, scale=self.scale)
        self.layer3 = block(in_channel, in_channel, kernel_size=3, dilation=4, scale=self.scale)
        self.layer4 = block(in_channel, in_channel, kernel_size=3, dilation=5, scale=self.scale)
        self.layer5 = nn.Conv1d(4*in_channel, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(1536, 384, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Conv1d(384, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.bn2 = nn.BatchNorm1d(3072)

        self.fc = nn.Linear(3072, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.layer5(torch.cat((x1, x2, x3, x4),dim=1))
        x = self.relu(x)


        w = self.attention(x)

        mean = torch.sum(x*w, dim=2)
        sigma = torch.sqrt(torch.sum((x**2)*w, dim=2)-mean**2)

        x = torch.cat((mean, sigma), 1)

        x = self.bn2(x)

        x = self.fc(x)

        x = self.bn3(x)

        return x