function output = CropImg(input, pad)

output = input(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2), :, :, :, :, :, :);