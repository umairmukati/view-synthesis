#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch.nn as nn
import pdb
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functions import get_variable, get_numpy
from importlib import reload, import_module

class ResidualNet(nn.Module):
    
    def __init__(self, input_shape, minibatch_size, lfsize, batchAffine = False, bias = False):
        super(ResidualNet, self).__init__()
        
        # net = Net((height, width), minibatch_size, patch_size, lfsize)
         
        network_file = r"network"
        model_file   = r"model\model.pt"
        
        network_module = import_module(network_file)
        reload(network_module)
        Net = network_module.Net

        self.vsnet = Net(input_shape, minibatch_size, lfsize, batchAffine=batchAffine)
        if torch.cuda.is_available():
            print('##converting vs network to cuda-enabled')
            self.vsnet.cuda()
            print('Model successfully loaded.')
            
        self.vsnet.eval()

        try:
            checkpoint = torch.load(model_file)

            for param in self.vsnet.parameters():
                param.require_grad = False

        except:
            print('No model.')
            
        
        # Residual CNN
        # 
        self.r_conv0 = nn.Conv2d(146, 128, kernel_size = 3, padding = 2, bias = bias,  dilation = 2)
        self.r_conv1 = nn.Conv2d(128, 128, kernel_size = 3, padding = 4, bias = bias,  dilation = 4)
        self.r_conv2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 8, bias = bias,  dilation = 8)
        self.r_conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 16, bias = bias, dilation = 16)
        self.r_conv4 = nn.Conv2d(128, 64,  kernel_size = 3, padding = 1, bias = bias)
        self.r_conv5 = nn.Conv2d(64,  64,  kernel_size = 3, padding = 1, bias = bias)
        self.r_conv6 = nn.Conv2d(64,  1,   kernel_size = 3, padding = 1, bias = bias)
        
        self.r_bn0 = nn.BatchNorm2d(128, affine=batchAffine)
        self.r_bn1 = nn.BatchNorm2d(128, affine=batchAffine)
        self.r_bn2 = nn.BatchNorm2d(128, affine=batchAffine)
        self.r_bn3 = nn.BatchNorm2d(128, affine=batchAffine)
        self.r_bn4 = nn.BatchNorm2d(64,  affine=batchAffine)
        self.r_bn5 = nn.BatchNorm2d(64,  affine=batchAffine)
        
      
    # Residual CNN
    def rcnn(self, x):
       
        x = self.r_bn0(F.elu(self.r_conv0(x)))
        x = self.r_bn1(F.elu(self.r_conv1(x)))
        x = self.r_bn2(F.elu(self.r_conv2(x)))
        x = self.r_bn3(F.elu(self.r_conv3(x)))
        x = self.r_bn4(F.elu(self.r_conv4(x)))
        x = self.r_bn5(F.elu(self.r_conv5(x)))
        x = torch.tanh(self.r_conv6(x))
        
        return x

    def forward(self, x, p, q):
        
        I, F_00, F_01, F_10, F_11, w_00, w_01, w_10, w_11, d, P, Q = self.vsnet(x, p, q)
        
        R_00 = (I - w_00)/2
        R_01 = (I - w_01)/2
        R_10 = (I - w_10)/2
        R_11 = (I - w_11)/2
                
        #R = torch.sum((I.unsqueeze(4)-W)*M*self.signR, dim = 4)
        
        R = self.rcnn(torch.cat((F_00, F_01, F_10, F_11, R_00, R_01, R_10, R_11, d, P, Q), 1))
        
        pdb.set_trace()
                
        return I, R
    
     
def img_diff(imgs):
    plt.imshow(np.abs(get_numpy((imgs[0][0]-imgs[1][0]).permute(1,2,0)))/2, vmin = 0.0, vmax = 0.02); plt.pause(1);
    
def img_show(img):
    plt.imshow(get_numpy(img[0].permute(1,2,0)+1)/2); plt.pause(1);

def img_disp(img):
    plt.imshow(get_numpy(img[0][0]+4)/8, cmap = 'gray', vmin = 0, vmax = 1.0); plt.pause(1);
