import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import numpy as np
import math
import cv2
import torch.nn.functional as F

class RIFNet(nn.Module):
    def __init__(self, load_weights=False):
        super(RIFNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat_1  = [512,512,512]
        self.backend_feat_2  = [256,128,64]

        self.middle_feat = [256,64,32]
        self.late_feat = [32]
        self.signal_feat  = [256,64,32]
        self.signal_out_feat = [32,1]
        
        self.frontend = make_layers(self.frontend_feat)
        self.backend_1 = make_layers(self.backend_feat_1,in_channels = 512,dilation = True)
        self.backend_2 = make_layers(self.backend_feat_2,in_channels = 512,dilation = True)
        self.middle_layer = make_layers(self.middle_feat,in_channels = 512,dilation = True)
        self.late_layer = make_layers(self.late_feat,in_channels = 64,dilation = True)
        self.signal_end = make_layers(self.signal_feat,in_channels = 1,dilation = True)
        self.signal_out = make_layers(self.signal_out_feat,in_channels = 32*3)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up = nn.Upsample((225,400),mode='bilinear')
        self.upsample = nn.Upsample(size=(225,400), mode='bilinear')
        
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]          
                
    def forward(self,img,sig_down):
        x = self.frontend(img)
        backend_1_output = self.backend_1(x)
        backend_2_output = self.backend_2(backend_1_output)
        density_map_up = self.output_layer(backend_2_output)
        density_map_up = self.up(density_map_up)
        
        middle = self.middle_layer(backend_1_output)
        late = self.late_layer(backend_2_output)
        sig = self.signal_end(sig_down)
        
        output = torch.cat((middle,late,sig),1)
        output = self.signal_out(output)
        output = self.upsample(output)
        
        out = torch.mul(density_map_up,output)
        
        return out       
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                