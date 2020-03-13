function createFlag = WriteTrainingExamples(inImgs, ref, refPos, outputDir, writeOrder, startInd, createFlag, arraySize)

chunkSize = 1000;
fileName = sprintf('%s/testing8.h5', outputDir);

[~, numElements] = size(refPos);


for k = 1 : numElements
    
    j = k + startInd - 1;
    
    curInImgs = inImgs(:, :, :, k);
    curRef = ref(:, :, :, k);
    curRefPos = refPos(:, k);
    
    SaveHDF(fileName, '/IN', single(curInImgs), PadWithOne(size(curInImgs), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
    SaveHDF(fileName, '/GT', single(curRef), PadWithOne(size(curRef), 4), [1, 1, 1, writeOrder(j)], chunkSize, createFlag, arraySize);
    SaveHDF(fileName, '/RP', single(curRefPos), size(curRefPos), [1, writeOrder(j)], chunkSize, createFlag, arraySize);
    
    createFlag = false;
end



