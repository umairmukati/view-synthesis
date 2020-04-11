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
        
        # net = Net((height, width), minibatch_size, patch_size, lfsize)
                
        self.input_channels = 3
        self.lfsize = lfsize
        self.dmax = 4
        nc_fh = nc_fd = 16
        nc_dh = nc_dd = 64
        nc_s = 128
        
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
        
        # For residual estimation
        self.signR = np.array([1, -1, 1, -1])
        self.signR = torch.from_numpy(self.signR[(None,)*4])
        
        # Converting to cuda tensor if available
        if torch.cuda.is_available():
            self.P = self.P.cuda()
            self.Q = self.Q.cuda()
            self.D = self.D.cuda()
            self.grid = self.grid.cuda()
            self.div = self.div.cuda()
            self.signR = self.signR.cuda()
    
        # Making beta learnable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta.requires_grad = True
        
        # Features Horizontal CNN
        # (input channels, output channels, kernel size, padding)
        self.fh_conv0 = nn.Conv2d(self.input_channels, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        self.fh_conv1 = nn.Conv2d(nc_fh, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        self.fh_conv2 = nn.Conv2d(nc_fh, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        self.fh_conv3 = nn.Conv2d(nc_fh, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        self.fh_conv4 = nn.Conv2d(nc_fh, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        self.fh_conv5 = nn.Conv2d(nc_fh*4, nc_fh, kernel_size = 3, padding = 1, bias = bias)
        
        self.fh_pool0 = nn.AvgPool2d(16, stride = 16)
        self.fh_pool1 = nn.AvgPool2d(8,  stride = 8)
        
        self.fh_bn0 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        self.fh_bn1 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        self.fh_bn2 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        self.fh_bn3 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        self.fh_bn4 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        self.fh_bn5 = nn.BatchNorm2d(nc_fh, affine=batchAffine)
        
        # Features Diagonal CNN
        # (input channels, output channels, kernel size, padding)
        self.fd_conv0 = nn.Conv2d(self.input_channels, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        self.fd_conv1 = nn.Conv2d(nc_fd, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        self.fd_conv2 = nn.Conv2d(nc_fd, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        self.fd_conv3 = nn.Conv2d(nc_fd, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        self.fd_conv4 = nn.Conv2d(nc_fd, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        self.fd_conv5 = nn.Conv2d(nc_fd*4, nc_fd, kernel_size = 3, padding = 1, bias = bias)
        
        self.fd_pool0 = nn.AvgPool2d(16, stride = 16)
        self.fd_pool1 = nn.AvgPool2d(8,  stride = 8)
        
        self.fd_bn0 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        self.fd_bn1 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        self.fd_bn2 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        self.fd_bn3 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        self.fd_bn4 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        self.fd_bn5 = nn.BatchNorm2d(nc_fd, affine=batchAffine)
        
        
        # Disparity Horizontal CNN
        # 
        self.dh_conv0 = nn.Conv2d(nc_dh//2, nc_dh, kernel_size = 3, padding = 2, bias = bias,  dilation = 2)
        self.dh_conv1 = nn.Conv2d(nc_dh, nc_dh, kernel_size = 3, padding = 4, bias = bias,  dilation = 4)
        self.dh_conv2 = nn.Conv2d(nc_dh, nc_dh, kernel_size = 3, padding = 8, bias = bias,  dilation = 8)
        self.dh_conv3 = nn.Conv2d(nc_dh, nc_dh, kernel_size = 3, padding = 16, bias = bias, dilation = 16)
        self.dh_conv4 = nn.Conv2d(nc_dh, nc_dh//2,  kernel_size = 3, padding = 1, bias = bias)
        self.dh_conv5 = nn.Conv2d(nc_dh//2, nc_dh//2,  kernel_size = 3, padding = 1, bias = bias)
        self.dh_conv6 = nn.Conv2d(nc_dh//2, 2,   kernel_size = 3, padding = 1, bias = bias)
        
        self.dh_bn0 = nn.BatchNorm2d(nc_dh, affine=batchAffine)
        self.dh_bn1 = nn.BatchNorm2d(nc_dh, affine=batchAffine)
        self.dh_bn2 = nn.BatchNorm2d(nc_dh, affine=batchAffine)
        self.dh_bn3 = nn.BatchNorm2d(nc_dh, affine=batchAffine)
        self.dh_bn4 = nn.BatchNorm2d(nc_dh//2,  affine=batchAffine)
        self.dh_bn5 = nn.BatchNorm2d(nc_dh//2,  affine=batchAffine)
        
        
        # Disparity Diagonal CNN
        # 
        self.dd_conv0 = nn.Conv2d(nc_dd//2, nc_dd, kernel_size = 3, padding = 2, bias = bias,  dilation = 2)
        self.dd_conv1 = nn.Conv2d(nc_dd, nc_dd, kernel_size = 3, padding = 4, bias = bias,  dilation = 4)
        self.dd_conv2 = nn.Conv2d(nc_dd, nc_dd, kernel_size = 3, padding = 8, bias = bias,  dilation = 8)
        self.dd_conv3 = nn.Conv2d(nc_dd, nc_dd, kernel_size = 3, padding = 16, bias = bias, dilation = 16)
        self.dd_conv4 = nn.Conv2d(nc_dd, nc_dd//2,  kernel_size = 3, padding = 1, bias = bias)
        self.dd_conv5 = nn.Conv2d(nc_dd//2, nc_dd//2,  kernel_size = 3, padding = 1, bias = bias)
        self.dd_conv6 = nn.Conv2d(nc_dd//2, 2,   kernel_size = 3, padding = 1, bias = bias)
        
        self.dd_bn0 = nn.BatchNorm2d(nc_dd, affine=batchAffine)
        self.dd_bn1 = nn.BatchNorm2d(nc_dd, affine=batchAffine)
        self.dd_bn2 = nn.BatchNorm2d(nc_dd, affine=batchAffine)
        self.dd_bn3 = nn.BatchNorm2d(nc_dd, affine=batchAffine)
        self.dd_bn4 = nn.BatchNorm2d(nc_dd//2,  affine=batchAffine)
        self.dd_bn5 = nn.BatchNorm2d(nc_dd//2,  affine=batchAffine)
        
        # Selection CNN
        # 
        self.s_conv0 = nn.Conv2d(50,  nc_s//2,  kernel_size = 3, padding = 1, bias = bias) 
        self.s_conv1 = nn.Conv2d(nc_s//2, nc_s, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv2 = nn.Conv2d(nc_s, nc_s, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv3 = nn.Conv2d(nc_s, nc_s, kernel_size = 3, padding = 1, bias = bias)
        self.s_conv4 = nn.Conv2d(nc_s, nc_s//2,  kernel_size = 3, padding = 1, bias = bias)
        self.s_conv5 = nn.Conv2d(nc_s//2,  nc_s//4,  kernel_size = 3, padding = 1, bias = bias)
        self.s_conv6 = nn.Conv2d(nc_s//4,  12,   kernel_size = 3, padding = 1, bias = bias)
        
        self.s_bn0 = nn.BatchNorm2d(nc_s//2,  affine=batchAffine)
        self.s_bn1 = nn.BatchNorm2d(nc_s, affine=batchAffine)
        self.s_bn2 = nn.BatchNorm2d(nc_s, affine=batchAffine)
        self.s_bn3 = nn.BatchNorm2d(nc_s, affine=batchAffine)
        self.s_bn4 = nn.BatchNorm2d(nc_s//2,  affine=batchAffine)
        self.s_bn5 = nn.BatchNorm2d(nc_s//4,  affine=batchAffine)
                
    # Features Horizontal CNN
    def fh_cnn(self, x):
       
        x = self.fh_bn0(F.elu(self.fh_conv0(x)))
        x = self.fh_bn1(F.elu(self.fh_conv1(x)))
        x = self.fh_bn2(F.elu(self.fh_conv2(x)))
        x_conv2 = x
        x = self.fh_bn3(F.elu(self.fh_conv3(x)))
        x = self.fh_bn4(F.elu(self.fh_conv4(x)))
        x_conv4 = x + x_conv2
        
        x_pool0 = F.upsample(self.fh_pool0(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        x_pool1 = F.upsample(self.fh_pool1(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        
        x = torch.cat((x_conv2, x_conv4, x_pool0, x_pool1), 1)
        
        x = self.fh_bn5(F.elu(self.fh_conv5(x)))
        
        return x
                
    # Features Diagonal CNN
    def fd_cnn(self, x):
       
        x = self.fd_bn0(F.elu(self.fd_conv0(x)))
        x = self.fd_bn1(F.elu(self.fd_conv1(x)))
        x = self.fd_bn2(F.elu(self.fd_conv2(x)))
        x_conv2 = x
        x = self.fd_bn3(F.elu(self.fd_conv3(x)))
        x = self.fd_bn4(F.elu(self.fd_conv4(x)))
        x_conv4 = x + x_conv2
        
        x_pool0 = F.upsample(self.fd_pool0(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        x_pool1 = F.upsample(self.fd_pool1(x_conv4), x_conv4.size()[2:4], 
                             mode='bilinear', align_corners = True)
        
        x = torch.cat((x_conv2, x_conv4, x_pool0, x_pool1), 1)
        
        x = self.fd_bn5(F.elu(self.fd_conv5(x)))
        
        return x
    
    # Disparity Horizontal CNN
    def dh_cnn(self, x):
       
        x = self.dh_bn0(F.elu(self.dh_conv0(x)))
        x = self.dh_bn1(F.elu(self.dh_conv1(x)))
        x = self.dh_bn2(F.elu(self.dh_conv2(x)))
        x = self.dh_bn3(F.elu(self.dh_conv3(x)))
        x = self.dh_bn4(F.elu(self.dh_conv4(x)))
        x = self.dh_bn5(F.elu(self.dh_conv5(x)))
        x = torch.tanh(self.dh_conv6(x))
        
        return x.mul(self.dmax)
    
    # Disparity Diagonal CNN
    def dd_cnn(self, x):
       
        x = self.dd_bn0(F.elu(self.dd_conv0(x)))
        x = self.dd_bn1(F.elu(self.dd_conv1(x)))
        x = self.dd_bn2(F.elu(self.dd_conv2(x)))
        x = self.dd_bn3(F.elu(self.dd_conv3(x)))
        x = self.dd_bn4(F.elu(self.dd_conv4(x)))
        x = self.dd_bn5(F.elu(self.dd_conv5(x)))
        x = torch.tanh(self.dd_conv6(x))
        
        return x.mul(self.dmax)
    
    # Selection CNN
    def s_cnn(self, x):
       
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
            
        # Top-Left (-3, -3),
        self.D[:,0,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) + self.lfsize[2] // 2
        self.D[:,0,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) + self.lfsize[2] // 2
        
        # Top-right (-3, +3),
        self.D[:,1,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) + self.lfsize[2] // 2
        self.D[:,1,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) - self.lfsize[2] // 2
        
        # Bottom-Left (+3, -3),
        self.D[:,2,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) - self.lfsize[2] // 2
        self.D[:,2,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) + self.lfsize[2] // 2
        
        # Bottom-right (+3, +3),
        self.D[:,3,:,:,1] = p[:,None,None]*(self.lfsize[2] // 2) - self.lfsize[2] // 2
        self.D[:,3,:,:,0] = q[:,None,None]*(self.lfsize[3] // 2) - self.lfsize[2] // 2
        
        self.P[:,:,:,:] = p[:,None,None,None]
        self.Q[:,:,:,:] = q[:,None,None,None]
        
        # [sample, channels, height, width, corner] 
        x_00 = x[:,:3]       # 0:3 top-left
        x_01 = x[:,6:9]      # 6:9 top-right
        x_10 = x[:,3:6]      # 3:6 bottom-left
        x_11 = x[:,9:]       # 9:12 bottom-right
        
        # Extracting horizontal, vertical and diagonal features
        FH_00 = self.fh_cnn(x_00); FHT_00 = self.fh_cnn(x_00.permute(0,1,3,2))
        FH_01 = self.fh_cnn(x_01); FHT_01 = self.fh_cnn(x_01.permute(0,1,3,2))
        FH_10 = self.fh_cnn(x_10); FHT_10 = self.fh_cnn(x_10.permute(0,1,3,2))
        FH_11 = self.fh_cnn(x_11); FHT_11 = self.fh_cnn(x_11.permute(0,1,3,2))
        
        FD_00 = self.fd_cnn(x_00)
        FDT_01 = self.fd_cnn(x_01.flip(2))
        FDT_10 = self.fd_cnn(x_10.flip(2))
        FD_11 = self.fd_cnn(x_11)
        
        # Estimating disparities
        d_TH = self.dh_cnn(torch.cat((FH_00, FH_01), 1))
        d_BH = self.dh_cnn(torch.cat((FH_10, FH_11), 1))
        d_LV = self.dh_cnn(torch.cat((FHT_00, FHT_10), 1)).permute(0,1,3,2)
        d_RV = self.dh_cnn(torch.cat((FHT_01, FHT_11), 1)).permute(0,1,3,2)
        
        d_BD = self.dd_cnn(torch.cat((FD_00, FD_11), 1))
        d_FD = self.dd_cnn(torch.cat((FDT_10, FDT_01), 1)).flip(2)
                
        Dh_00 = d_TH[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,0]
        Dh_01 = d_TH[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,1]
        Dh_10 = d_BH[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,2]
        Dh_11 = d_BH[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,3]
        
        Dv_00 = d_LV[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,0]
        Dv_01 = d_RV[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,1]
        Dv_10 = d_LV[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,2]
        Dv_11 = d_RV[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,3]
        
        Dd_00 = d_BD[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,0]
        Dd_01 = d_FD[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,1]
        Dd_10 = d_FD[:,0,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,2]
        Dd_11 = d_BD[:,1,:,:].unsqueeze(3).repeat(1,1,1,2)*self.D[:,3]
        
        # Warping views
        
        Wh_00 = F.grid_sample(x_00, self.grid + Dh_00/(self.div), align_corners=False)
        Wh_01 = F.grid_sample(x_01, self.grid + Dh_01/(self.div), align_corners=False)
        Wh_10 = F.grid_sample(x_10, self.grid + Dh_10/(self.div), align_corners=False)
        Wh_11 = F.grid_sample(x_11, self.grid + Dh_11/(self.div), align_corners=False)
        
        Wv_00 = F.grid_sample(x_00, self.grid + Dv_00/(self.div), align_corners=False)
        Wv_01 = F.grid_sample(x_01, self.grid + Dv_01/(self.div), align_corners=False)
        Wv_10 = F.grid_sample(x_10, self.grid + Dv_10/(self.div), align_corners=False)
        Wv_11 = F.grid_sample(x_11, self.grid + Dv_11/(self.div), align_corners=False)
        
        Wd_00 = F.grid_sample(x_00, self.grid + Dd_00/(self.div), align_corners=False)
        Wd_01 = F.grid_sample(x_01, self.grid + Dd_01/(self.div), align_corners=False)
        Wd_10 = F.grid_sample(x_10, self.grid + Dd_10/(self.div), align_corners=False)
        Wd_11 = F.grid_sample(x_11, self.grid + Dd_11/(self.div), align_corners=False)
        
        W = torch.cat((Wh_00[:,:,:,:,None], Wh_01[:,:,:,:,None], Wh_10[:,:,:,:,None], Wh_11[:,:,:,:,None], 
                       Wv_00[:,:,:,:,None], Wv_01[:,:,:,:,None], Wv_10[:,:,:,:,None], Wv_11[:,:,:,:,None], 
                       Wd_00[:,:,:,:,None], Wd_01[:,:,:,:,None], Wd_10[:,:,:,:,None], Wd_11[:,:,:,:,None]),4)
                
        M = self.s_cnn(torch.cat((Wh_00, Wh_01, Wh_10, Wh_11, \
                                  Wv_00, Wv_01, Wv_10, Wv_11, \
                                  Wd_00, Wd_01, Wd_10, Wd_11, \
                                  d_TH, d_BH, d_LV, d_RV, d_BD, d_FD, \
                                  self.P, self.Q), 1))
        M = M.unsqueeze(4).permute(0, 4, 2, 3, 1)
        
        I = torch.sum(M*W, dim = 4)
                        
        #R = torch.sum((I.unsqueeze(4)-W)*M*self.signR, dim = 4)
        
        return I, M
    
    
def img_diff(imgs):
    plt.imshow(np.abs(get_numpy((imgs[0][0]-imgs[1][0]).permute(1,2,0)))/2, vmin = 0.0, vmax = 0.02); plt.pause(1);
    
def img_show(img):
    plt.imshow(get_numpy(img[0].permute(1,2,0)+1)/2); plt.pause(1);

def img_disp(img):
    plt.imshow(get_numpy(img[0][0]+4)/8, cmap = 'gray', vmin = 0, vmax = 1.0); plt.pause(1);
