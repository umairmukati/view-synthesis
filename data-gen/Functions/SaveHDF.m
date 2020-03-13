function startLoc = SaveHDF(fileName, datasetName, input, inDims, startLoc, chunkSize, createFlag, arraySize)

if (~exist('arraySize', 'var') || isempty(arraySize))
    arraySize = inf;
end

if (createFlag)
    h5create(fileName, datasetName, [inDims(1:end-1), arraySize], 'Datatype', 'uint8', 'ChunkSize', [inDims(1:end-1), chunkSize]);
end

h5write(fileName, datasetName, uint8(round(input*255)), startLoc, inDims);

startLoc(end) = startLoc(end) + inDims(end);

