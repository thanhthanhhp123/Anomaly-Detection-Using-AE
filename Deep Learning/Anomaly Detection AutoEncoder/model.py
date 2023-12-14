import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 48, 11, 1, 5)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(48, 48, 9, 2, 4)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 48, 7, 2, 3)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 48, 5, 2, 2)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(48, 48, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(48)

        self.conv6 = nn.ConvTranspose2d(48, 48, 5, 2, 2, output_padding=1)
        self.bn6 = nn.BatchNorm2d(48)

        self.conv7 = nn.ConvTranspose2d(96, 48, 7, 2, 3, output_padding=1)
        self.bn7 = nn.BatchNorm2d(48)

        self.conv8 = nn.ConvTranspose2d(96, 48, 9, 2, 4, output_padding=1)
        self.bn8 = nn.BatchNorm2d(48)

        self.conv9 = nn.ConvTranspose2d(96, 48, 11, 2, 5, output_padding=1)
        self.bn9 = nn.BatchNorm2d(48)

        self.out = nn.Conv2d(48, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))

        x6 = F.relu(self.bn6(self.conv6(x5)))
        x7 = F.relu(self.bn7(self.conv7(torch.cat([x6, x4], 1))))
        x8 = F.relu(self.bn8(self.conv8(torch.cat([x7, x3], 1))))
        x9 = F.relu(self.bn9(self.conv9(torch.cat([x8, x2], 1))))

        x10 = self.bn(self.out(x9))
        return x10

if __name__ == '__main__':
    from torchsummary import summary
    model = AE(3)
    summary(model, (3, 256, 256))