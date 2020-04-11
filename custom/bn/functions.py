import torch
import h5py as h5
import numpy as np
import torch.nn.functional as F
import math

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, archive, lfsize, transform=None):
        self.lfsize = lfsize
        self.archive = h5.File(archive, 'r')
        self.target = self.archive['GT']
        self.data = self.archive['IN']
        self.labels = self.archive['RP']

        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]

        labels = ((self.labels[index].astype('float')-1)-self.lfsize[2]//2)/(self.lfsize[2]//2)
        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        return data, target, labels

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archive.close()

def customTransform(data, gamma_val = 0.4):
    return 2 * torch.pow(data.permute(1,0,2), gamma_val) - 1

def compute_gradient(tensor):
    sobel_x = torch.Tensor(np.array([[[[-1., 0, 1.],
                                       [-2., 0, 2.],
                                       [-1., 0, 1.]]]], dtype=np.float32))
    sobel_y = torch.Tensor(np.array([[[[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]]]], dtype=np.float32))
    
    
    if torch.cuda.is_available():
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
    
    n,c,h,w=tensor.shape
    
    gradient_x = F.conv2d(tensor.reshape(n*c,1,h,w), sobel_x)
    gradient_y = F.conv2d(tensor.reshape(n*c,1,h,w), sobel_y)
    
    h1=gradient_x.shape[2]
    w1=gradient_x.shape[3]
    
    return torch.cat([gradient_x.reshape(n,c,h1,w1), gradient_y.reshape(n,c,h1,w1)], dim=1)

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()

def psnr_1(img1, img2):
  
    mse = np.mean( ((img1 - img2)/2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 10 * math.log10(PIXEL_MAX / mse)

def processLF(lf, lfsize, gamma_val):
    
    # Crop image
    lf = lf[:3,:lfsize[0]*14,:lfsize[1]*14]
    # 2D lenslet grid to 4D
    lf = lf.view(3, lfsize[0], 14, lfsize[1], 14).permute(1, 3, 2, 4, 0)
    # Pick the central perspectives only
    lf = lf[:, :, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2) + 1, (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2) + 1, :]
    # Gamma correction
    lf = torch.pow(lf, gamma_val)
    # Normalize LF (-1 to 1)
    lf = (lf * 2.0) - 1.0
    
    return lf