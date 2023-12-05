import torch.nn as nn
import torch

class CSAM(nn.Module):
    def __init__(self, in_planes,kernel_size=7):
        super(CSAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out1 = self.sigmoid(max_out)
        max_out, _ = torch.max(out1, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class FAD(nn.Module):
    def __init__(self,Channel_S,Channel_T,TransToChannel):
        super(FAD, self).__init__()
        self.MSE = nn.MSELoss(reduction='mean')
        self.TransToChannelTeacher = nn.Conv2d(Channel_T, TransToChannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.TransToChannelStudent = nn.Conv2d(Channel_S, TransToChannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.casm1=CSAM(in_planes=Channel_S)
        self.casm2=CSAM(in_planes=Channel_T)
    def forward(self, Frj,Fdj,frj,fdj):
        F_fused=Frj+Fdj
        f_fused=frj+fdj
        F_MDR=self.casm2(self.TransToChannelTeacher(F_fused))
        f_MDR=self.casm1(self.TransToChannelStudent(f_fused))
        loss=self.MSE(F_MDR,f_MDR)
        return loss
