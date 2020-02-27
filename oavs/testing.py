import numpy as np
import h5py as h5

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from importlib import reload, import_module

import glob
import os

import pdb
from PIL import Image as im
import _pickle as pickle
import cv2

from functions import processLF, get_variable, get_numpy, psnr_1

class test():
    
    def __init__(self):
        
        self.lfsize         = [372, 540, 7, 7]
        self.batch_affine   = True

    def createNet(network_file = r"network"):

        network_module = import_module(network_file)
        reload(network_module)

        Net = network_module.Net

        net = Net((lfsize[0], lfsize[1]), 1, self.lfsize, batchAffine=self.batch_affine)
        net.eval()

        if torch.cuda.is_available():
            print('##converting network to cuda-enabled')
            net.cuda()

        try:
            checkpoint = torch.load(model_file)

            net.load_state_dict(checkpoint['model'].state_dict())    
            print('Model successfully loaded.')

        except:
            print('No model.')

        return net

    def synthesizeView(corn, index):

        p = np.ndarray([1])
        q = np.ndarray([1])

        p[0] = (index[0] - self.lfsize[2]//2)/(self.lfsize[2]//2)
        q[0] = (index[1] - self.lfsize[3]//2)/(self.lfsize[3]//2)

        corn = corn.permute(2,3,0,1).reshape(12,corn.shape[0],corn.shape[1])[None,:]

        with torch.no_grad():
            Y, R = net(get_variable(corn), get_variable(torch.from_numpy(p)), get_variable(torch.from_numpy(q)))

        return Y[0].permute(1,2,0), R[0].permute(1,2,0)


