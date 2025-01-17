function output = AdjustTone(input)

input(input > 1) = 1;
input(input < 0) = 0;

output = input.^(1/1.5);
output = rgb2hsv(output);
output(:, :, 2) = output(:, :, 2) * 1.5;
output = hsv2rgb(output);