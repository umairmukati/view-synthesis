%pyversion('/appl/python/3.6.2/bin/python3.6')

imgdir = '/zhome/e1/5/136113/Desktop/Datasets/DVC/10b/gray/orig/Bikes/';

ctl = rgb2gray(double(imread(strcat(imgdir,'002_002.ppm')))/65535)*2 - 1;
ctr = rgb2gray(double(imread(strcat(imgdir,'008_002.ppm')))/65535)*2 - 1;
cbl = rgb2gray(double(imread(strcat(imgdir,'002_008.ppm')))/65535)*2 - 1;
cbr = rgb2gray(double(imread(strcat(imgdir,'008_008.ppm')))/65535)*2 - 1;

mod.torch = py.importlib.import_module('torch');
mod.numpy = py.importlib.import_module('numpy');
mod.testing = py.importlib.import_module('testing');

mypy = py.testing.test(py.tuple(uint16([size(ctl), 7, 7])));

mypy.createNet()

[Y, R, D] = synthesizeView(ctl, cbl, ctr, cbr, [3, 3], mypy);