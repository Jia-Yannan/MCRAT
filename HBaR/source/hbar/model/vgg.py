import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, **kwargs):
        super(VGG16, self).__init__()

        self.rob = kwargs['robustness'] if 'robustness' in kwargs else False
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, 10)
        
        self.reshape = torch.nn.Sequential(
            nn.Linear(4096, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, reg_feature_dim, bias=True)
        )
        
        
    def forward(self, x, mcrat=0):
        output_list = []
            
        out = self.layer1(x)
        output_list.append(out)
        
        out = self.layer2(out)
        output_list.append(out)
        
        out = self.layer3(out)
        output_list.append(out)
        
        out = self.layer4(out)
        output_list.append(out)
        
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        
        #print(out.shape)
        out = self.layer6(out)
        output_list.append(out)
        
        outL = self.layer7(out)
        output_list.append(out)
        
        out = self.layer8(outL)
        
        if mcrat==1:
            #--
            outF = self.reshape(outL)
            outF = F.normalize(outF)
            #--
            if self.rob:
                return out, outF
            else:
                return out, output_list, outF
        else:
            if self.rob:
                return out
            else:
                return out, output_list

