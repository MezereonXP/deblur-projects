import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)   
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv11 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1) 
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1) 
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)           
        
    def forward(self, x):
        tmp = self.conv1(x)
        x1 = F.relu(self.conv2(tmp))
        x1 = self.conv3(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv4(tmp))
        x1 = self.conv5(x1)
        x1 = x1 + tmp  # residual link
        tmp = self.conv6(x1)
        x1 = F.relu(self.conv7(tmp))
        x1 = self.conv8(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv9(tmp))
        x1 = self.conv10(x1)
        x1 = x1 + tmp  # residual link
        tmp = self.conv11(x1)
        x1 = F.relu(self.conv12(tmp))
        x1 = self.conv13(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv14(tmp))
        x1 = self.conv15(x1)
        x1 = x1 + tmp  # residual link
        return x1