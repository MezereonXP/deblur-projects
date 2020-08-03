import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2)
        self.conv9 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.conv13 = nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        tmp = x
        x1 = F.relu(self.conv1(tmp))
        x1 = self.conv2(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv3(tmp))
        x1 = self.conv4(x1)
        x1 = x1 + tmp  # residual link
        tmp = self.deconv1(x1)
        x1 = F.relu(self.conv5(tmp))
        x1 = self.conv6(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv7(tmp))
        x1 = self.conv8(x1)
        x1 = x1 + tmp  # residual link
        tmp = self.deconv2(x1)
        x1 = F.relu(self.conv9(tmp))        
        x1 = self.conv10(x1)
        tmp = x1 + tmp  # residual link
        x1 = F.relu(self.conv11(tmp))
        x1 = self.conv12(x1)
        x1 = x1 + tmp  # residual link
        return self.conv13(x1)