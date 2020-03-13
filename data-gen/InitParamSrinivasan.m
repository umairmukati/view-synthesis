function InitParamSrinivasan()

global param;

%%% DO NOT CHANGE ANY PARAMETER, UNLESS YOU KNOW EXACTLY WHAT EACH PARAMETER DOES

%param.depthResolution = 100; % number of depth levels (see Eq. 5)
%param.numDepthFeatureChannels = param.depthResolution * 2;
%param.deltaDisparity = 21; % the range of disparities (see Eq. 5)
param.origAngRes = 7; % original angular resolution
%param.depthBorder = 6; % the number of pixels that are reduce due to the convolution process in the depth network. This border can be avoided if the networks are padded appropriately.
%param.colorBorder = 6;  % same as above, for the color network.
%param.testNet = 'TrainedNetworks';
%param.gammaValue = 0.4;

%%% here, we set the desired novel views and indexes of the input views.
global novelView;
global inputView;

[novelView.X, novelView.Y] = meshgrid(linspace(-1,1,param.origAngRes), linspace(-1,1,param.origAngRes));
[inputView.X, inputView.Y] = meshgrid([-1,1], [-1,1]);
novelView.Y = novelView.Y'; novelView.X = novelView.X';
novelView.Y = novelView.Y(:); novelView.X = novelView.X(:);
inputView.Y = inputView.Y(:); inputView.X = inputView.X(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% If you have compiled MatConvNet with GPU and CuDNN supports, then leave
%%% these parameters as is. Otherwise change them appropriately.
%param.useGPU = true; 
%param.gpuMethod = 'Cudnn';%'NoCudnn';%


%%%%%%%%%%%%%%%%%% Training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.height = 376;
param.width = 541;

param.patchSize = 192;
param.stride = 96;
param.numRefs = 8; % number of reference images selected randomly on each light field
param.numRefsTest = param.origAngRes.^2 - 4;
param.cropSizeTraining = [25 90]; % we crop the boundaries to avoid artifacts in the training
param.gamma = 1;
%param.batchSize = 10; % we found batch size of 10 is faster and gives the same results

param.cropHeight = param.height - 2 * param.cropSizeTraining(1);
param.cropWidth = param.width - 2 * param.cropSizeTraining(2);

param.trainingScenes = 'C:\Users\mummu\Documents\Datasets\Srinivasan\Flowers_8bit\TestSet\';
param.trainingData = 'C:\Users\mummu\Documents\Datasets\Srinivasan\Flowers_8bit\TestSet\Data\';
[~, param.trainingNames, ~] = GetFolderContent(param.trainingData, '.h5');

param.testScenes = 'C:\Users\mummu\Documents\Datasets\Srinivasan\Flowers_8bit\TestSet\';
param.testData = 'C:\Users\mummu\Documents\Datasets\Srinivasan\Flowers_8bit\TestSet\Data\';
[~, param.testNames, ~] = GetFolderContent(param.testData, '.h5');

%param.trainNet = 'TrainingData';


%param.continue = true;
%param.startIter = 0;

%param.testNetIter = 100;
%param.printInfoIter = 5;


%%% ADAM parameters
%param.alpha = 0.0001;
%param.beta1 = 0.9;
%param.beta2 = 0.999;
%param.eps = 1e-8;