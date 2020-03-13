function createFlag = WriteTestExamples(inImgs, ref, refPos, outputDir, startInd, createFlag, arraySize)

chunkSize = 1;
fileName = sprintf('%s\\testing8.h5', outputDir);

SaveHDF(fileName, '/IN', single(inImgs), PadWithOne(size(inImgs), 4), [1, 1, 1, startInd], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/GT', single(ref), PadWithOne(size(ref), 4), [1, 1, 1, startInd], chunkSize, createFlag, arraySize);
SaveHDF(fileName, '/RP', single(refPos), size(refPos), [1, startInd], chunkSize, createFlag, arraySize);

createFlag = false;