function [inImgs, ref, refPos] = ComputeTestExamples(curFullLF, curInputLF)

global param;
numRefs = param.numRefsTest;
origAngRes = param.origAngRes;

[height, width, ~, ~, ~] = size(curInputLF);

%%%%%%%%%%%%% preparing input images %%%%%%%%%%%%%%%
inImgs = reshape(curInputLF, height, width, []);

%%%%%%%%%%%%% selecting random references %%%%%%%%%%
%numSeq = origAngRes^2;
refInds = 1:origAngRes^2;
refInds(any(refInds == repmat([1,origAngRes,origAngRes^2 - origAngRes + 1, origAngRes^2]',[1 origAngRes^2]))) = [];
ref = zeros(height, width, 3, numRefs);

refPos = zeros(2, numRefs);

for ri = 1 : numRefs
    
    [curRefInd.Y, curRefInd.X] = ind2sub([origAngRes, origAngRes], refInds(ri));
    
    fprintf('Working on reference %d of %d: ', ri, numRefs);
    
    %%%%%%%%%%%%%%%%%%%%% preparing reference %%%%%%%%%%%%%%%%%%%%%%%%%%%
    ref(:, :, :, ri) = curFullLF(:, :, :, curRefInd.Y, curRefInd.X);

    %%%%%%%%%%%%%%%%%%%%%% preparing ref positions %%%%%%%%%%%%%%%%%%%
    refPos(1, ri) = curRefInd.Y/255;
    refPos(2, ri) = curRefInd.X/255;
    
    fprintf(repmat('\b', 1, 5));
    fprintf('Done\n');
end

ref = reshape(ref, height, width, []);


