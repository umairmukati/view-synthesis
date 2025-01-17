import numpy as np
import torch
from importlib import reload, import_module
import pdb

from functions import get_variable

class test():
    
    def __init__(self, lfsize = [372, 540, 7, 7]):
        
        self.lfsize         = lfsize
        self.batch_affine   = True

    def createNet(self, network_file = 'network', model_file = 'model/model.pt'):

        network_module = import_module(network_file)
        reload(network_module)
        
        Net = network_module.Net

        self.net = Net((self.lfsize[0], self.lfsize[1]), 1, self.lfsize, batchAffine=self.batch_affine)
        self.net.eval()

        if torch.cuda.is_available():
            print('##converting network to cuda-enabled')
            self.net.cuda()

        try:
            checkpoint = torch.load(model_file)

            self.net.load_state_dict(checkpoint['model'].state_dict())    
            print('Model successfully loaded.')

        except:
            print('No model.')

    def synthesizeView(self, corn, index):
                
        p = np.ndarray([1])
        q = np.ndarray([1])

        p[0] = (index[0] - self.lfsize[2]//2)/(self.lfsize[2]//2)
        q[0] = (index[1] - self.lfsize[3]//2)/(self.lfsize[3]//2)

        corn = corn.permute(2,3,0,1).reshape(12,corn.shape[0],corn.shape[1])[None,:]

        with torch.no_grad():
            Y, R, d = self.net(get_variable(corn), get_variable(torch.from_numpy(p)), get_variable(torch.from_numpy(q)))

        return Y[0].permute(1,2,0), R[0].permute(1,2,0), d[0]


