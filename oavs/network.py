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

class Net(nn.Module):
    
    def __init__(self, input_shape, minibatch_size, lfsize, batchAffine = False, bias = False):
        super(Net, self).__init__()
        
        
        #pdb.set_trace()
        # net = Net((height, width), minibatch_size, patch_size, lfsize)
                
        self.input_channels = 5
        self.lfsize = lfsize
        self.dmax = 4
        
        # Matrix P and Q will be passed to the network
        self.P = torch.empty((minibatch_size,) + input_shape).unsqueeze(1)
        self.Q = torch.empty((minibatch_size,) + input_shape).unsqueeze(1)
        
        # Memory to hold multiplier for correct disparity
        # 4 represents four disparity maps
        self.D = torch.empty(minibatch_size,4,1,1,2)
        
        # A divider to normalize the range of disparities using the image size
        self.div = torch.from_numpy(np.array(input_shape[1::-1])[None,None,None,:]/2).float()
        
        # Grid to warp image
        self.grid_w, self.grid_h = np.meshgrid(np.linspace(-1, 1, input_shape[1]),
                                               np.linspace(-1, 1, input_shape[0]))
        
        self.grid = torch.stack((
            torch.tensor(self.grid_w, dtype=torch.float32), 
            torch.tensor(self.grid_h, dtype=torch.float32)),2).unsqueeze(0)
        
        # Converting to cuda tensor if available
        if torch.cuda.is_available():
            self.P = self.P.cuda()
            self.Q = self.Q.cuda()
            self.D = self.D.cuda()
            self.grid = self.grid.cuda()
            self.div = self.div.cuda()
    
        # Making beta learnable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta.requires_grad = True
        
        # Features CNN
        # (input channels, output channels, kernel size, padding)
        self.f_conv0 = nn.Conv2d(self.input_channels, 32, kernel_size = 3, padding = 1, bias = bias)
        self.f_conv1 = nn.Conv2d(32,  32, kernel_size = 3, padding = 1, bias = bias)
        self.f_conv2 = nn.Conv2d(32,  32, kernel_size = 3, padding = 1, bias = bias)
        self.f_conv3 = nn.Conv2d(32,  32, kernel_size = 3, padding = 1, bias = bias)
        self.f_conv4 = nn.Conv2d(32,  32, kernel_size = 3, padding = 1, bias = bias)
        
        self.f_pool0 = nn.AvgPool2d(16, stride = 16)
        self.f_pool1 = nn.AvgPool2d(8,  stride = 8)
        
        self.f_conv5 = nn.Conv2d(128, 32, kernel_size = 3, padding = 1, bias = bias)
        
        self.f_bn0 = nn.BatchNorm2d(32, affine=batchAffine)
        self.f_bn1 = nn.BatchNorm2d(32, affine=batchAffine)
        self.f_bn2 = nn.BatchNorm2d(32, affine=batchAffine)
        self.f_bn3 = nn.BatchNorm2d(32, affine=batchAffine)
        self.f_bn4 = nn.BatchNorm2d(32, affine=batchAffine)
        self.f_bn5 = nn.BatchNorm2d(32, affine=batchAffine)
        
        
        # Disparity CNN
        # 
        self.d_conv0 = nn.Conv2d(130, 128, kernel_size = 3, padding = 2, bias = bias,  dilation = 2)
        self.d_conv1 = nn.Conv2d(128, 128, kernel_size = 3, padding = 4, bias = bias,  dilation = 4)
        self.d_conv2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 8, bias = bias,  dilation = 8)
        self.d_conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 16, bias = bias, dilation = 16)
        self.d_conv4 = nn.Conv2d(128, 64,  kernel_size = 3, padding = 1, bias = bias)
        self.d_conv5 = nn.Conv2d(64,  64,  kernel_size = 3, padding = 1, bias = bias)
        self.d_conv6 = nn.Conv2d(64,  4,   kernel_size = 3, padding = 1, bias = bias)
        
        self.d_bn0 = nn.BatchNorm2d(128, affine=batchAffine)
        self.d_bn1 = nn.BatchNorm2d(128, affine=batchAffine)
        self.d_bn2 = nn.BatchNorm2d(128, affine=batchAffine)
        self.d_bn3 = nn.BatchNorm2d(128, affine=batchAffine)
        self.d_bn4 = nn.BatchNorm2d(64,  affine=batchAffine)
        self.d_bn5 = nn.BatchNorm2d(64,  affine=batchAffine)
        
        # Selection CNN
        # 
        self.s_conv0 = nn.Conv2d(18,  64,  kernel_size = 3, padding = 1, bias = bias) 
        self.s_conv1 = nn.Conv2d(64,  128, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv4 = nn.Conv2d(128, 64,  kernel_size = 3, padding = 1, bias = bias)
        self.s_conv5 = nn.Conv2d(64,  32,  kernel_size = 3, padding = 1, bias = bias)
        self.s_conv6 = nn.Conv2d(32,  4,   kernel_size = 3, padding = 1, bias = bias)
        
        self.s_bn0 = nn.BatchNorm2d(64,  affine=batchAffine)
        self.s_bn1 = nn.BatchNorm2d(128, affine=batchAffine)
        self.s_bn2 = nn.BatchNorm2d(128, affine=batchAffine)
        self.s_bn3 = nn.BatchNorm2d(128, affine=batchAffine)
        self.s_bn4 = nn.BatchNorm2d(64,  affine=batchAffine)
        self.s_bn5 = nn.BatchNorm2d(32,  affine=batchAffine)
                
    # Features CNN
    def fcnn(self, x):
       
        x = self.f_bn0(F.elu(self.f_conv0(x)))
        x = self.f_bn1(F.elu(self.f_conv1(x)))
        x = self.f_bn2(F.elu(self.f_conv2(x)))
        x_conv2 = x
        x = self.f_bn3(F.elu(self.f_conv3(x)))
        x = self.f_bn4(F.elu(self.f_conv4(x)))
        x_conv4 = x + x_conv2
        
        x_pool0 = F.upsample(self.f_pool0(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        x_pool1 = F.upsample(self.f_pool1(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        
        x = torch.cat((x_conv2, x_conv4, x_pool0, x_pool1), 1)
        
        x = self.f_bn5(F.elu(self.f_conv5(x)))
        
        return x
    
    # Disparity CNN
    def dcnn(self, x):
       
        x = self.d_bn0(F.elu(self.d_conv0(x)))
        x = self.d_bn1(F.elu(self.d_conv1(x)))
        x = self.d_bn2(F.elu(self.d_conv2(x)))
        x = self.d_bn3(F.elu(self.d_conv3(x)))
        x = self.d_bn4(F.elu(self.d_conv4(x)))
        x = self.d_bn5(F.elu(self.d_conv5(x)))
        x = torch.tanh(self.d_conv6(x))
        
        return x.mul(self.dmax)
    
    # Selection CNN
    def scnn(self, x):
       
        x = self.s_bn0(F.elu(self.s_conv0(x)))
        x = self.s_bn1(F.elu(self.s_conv1(x)))
        x = self.s_bn2(F.elu(self.s_conv2(x)))
        x = self.s_bn3(F.elu(self.s_conv3(x)))
        x = self.s_bn4(F.elu(self.s_conv4(x)))
        x = self.s_bn5(F.elu(self.s_conv5(x)))
        x = torch.tanh(self.s_conv6(x))
        
        x = F.softmax(self.beta*x,dim = 1)
        
        return x

    def forward(self, x, p, q):
        # extract input features
                          
        #pdb.set_trace()
            
        # Top-Left (-3, -3),
        self.D[:,0,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) + 3
        self.D[:,0,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) + 3
        
        # Top-right (-3, +3),
        self.D[:,1,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) + 3
        self.D[:,1,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) - 3
        
        # Bottom-Left (+3, -3),
        self.D[:,2,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) - 3
        self.D[:,2,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) + 3
        
        # Bottom-right (+3, +3),
        self.D[:,3,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) - 3
        self.D[:,3,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) - 3
        
        self.P[:,:,:,:] = p[:,None,None,None]
        self.Q[:,:,:,:] = q[:,None,None,None]
        
        # [sample, channels, height, width, corner] 
        x_00 = x[:,:3]       # 0:3 top-left
        x_01 = x[:,6:9]      # 6:9 top-right
        x_10 = x[:,3:6]      # 3:6 bottom-left
        x_11 = x[:,9:]       # 9:12 bottom-right
        
        F_00 = self.fcnn(torch.cat((x_00, self.P, self.Q), 1))
        F_01 = self.fcnn(torch.cat((x_01, self.P, self.Q), 1))
        F_10 = self.fcnn(torch.cat((x_10, self.P, self.Q), 1))
        F_11 = self.fcnn(torch.cat((x_11, self.P, self.Q), 1))
        
        
        # input_channels = 32 + 32 + 32 + 32 + 1 + 1
        d = self.dcnn(torch.cat((F_00, F_01, F_10, F_11, self.P, self.Q), 1))
                
        D_00 = d[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,0]
        D_01 = d[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,1]
        D_10 = d[:,2,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,2]
        D_11 = d[:,3,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,3]
        
        w_00 = F.grid_sample(x_00, self.grid + D_00/(self.div), align_corners=False)
        w_01 = F.grid_sample(x_01, self.grid + D_01/(self.div), align_corners=False)
        w_10 = F.grid_sample(x_10, self.grid + D_10/(self.div), align_corners=False)
        w_11 = F.grid_sample(x_11, self.grid + D_11/(self.div), align_corners=False)
        
        W = torch.cat((w_00[:,:,:,:,None], w_01[:,:,:,:,None], 
                       w_10[:,:,:,:,None], w_11[:,:,:,:,None]),4)
        W = W.permute(0,4,2,3,1)
        
        M = self.scnn(torch.cat((w_00, w_01, w_10, w_11, d, self.P, self.Q), 1))
        
        
        I = torch.sum(M.unsqueeze(4)*W, dim = 1).permute(0, 3, 1, 2)
        
        #pdb.set_trace()
                
        return I
