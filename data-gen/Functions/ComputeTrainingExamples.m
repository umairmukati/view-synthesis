function [pInImgs, pRef, refPos] = ComputeTrainingExamples(curFullLF, curInputLF)

global param;
cropSize = param.cropSizeTraining;
numRefs = param.numRefs;
patchSize = param.patchSize;
stride = param.stride;
origAngRes = param.origAngRes;


%%%%%%%%%%%%% preparing input images %%%%%%%%%%%%%%%
[height, width, ~, ~, ~] = size(curInputLF);
inImgs = reshape(curInputLF, height, width, []);
inImgs = CropImg(inImgs, cropSize);
pInImgs = GetPatches(inImgs, patchSize, stride);
pInImgs = repmat(pInImgs, [1, 1, 1, numRefs]);


%%%%%%%%%%%%% selecting random references %%%%%%%%%%
numSeq = randperm(origAngRes^2);
%numSeq(any(numSeq == repmat([1,origAngRes,origAngRes^2 - origAngRes + 1, origAngRes^2]',[1 origAngRes^2]))) = [];
refInds = numSeq(1:numRefs);

%%%%%%%%%%%%% initializing the arrays %%%%%%%%%%%%%%
numPatches = GetNumPatches();
pRef = zeros(patchSize, patchSize, 3, numPatches * numRefs);
refPos = zeros(2, numPatches * numRefs);

for ri = 1 : numRefs
    
    fprintf('Working on random reference %d of %d: ', ri, numRefs);
    
    [curRefInd.Y, curRefInd.X] = ind2sub([origAngRes, origAngRes], refInds(ri));
    curRefPos.Y = GetImgPos(curRefInd.Y); curRefPos.X = GetImgPos(curRefInd.X);
    
    wInds = (ri-1) * numPatches + 1 : ri * numPatches;
    
    %%%%%%%%%%%%%%%%%%%%% preparing reference %%%%%%%%%%%%%%%%%%%%%%%%%%%
    ref = curFullLF(:, :, :, curRefInd.Y, curRefInd.X);
    ref = CropImg(ref, cropSize);
    pRef(:, :, :, wInds) = GetPatches(ref, patchSize, stride);
    
    
    %%%%%%%%%%%%%%%%%%%%% preparing features %%%%%%%%%%%%%%%%%%%%%%%%%
    %deltaViewY = inputView.Y - curRefPos.Y; 
    %deltaViewX = inputView.X - curRefPos.X;

    %inFeat = PrepareDepthFeatures(curInputLF, deltaViewY, deltaViewX);
    %inFeat = CropImg(curInputLF, cropSize);
    %pInFeat(:, :, :, wInds) = GetPatches(inFeat, patchSize, stride);
    
    
    %%%%%%%%%%%%%%%%%%%%%% preparing ref positions %%%%%%%%%%%%%%%%%%%
    refPos(1, wInds) = repmat(curRefInd.Y/255, [1, numPatches]);
    refPos(2, wInds) = repmat(curRefInd.X/255, [1, numPatches]);
    
    fprintf(repmat('\b', 1, 5));
    fprintf('Done\n');
end



