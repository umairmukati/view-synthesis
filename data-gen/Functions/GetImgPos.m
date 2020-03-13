function pos = GetImgPos(ind)

global param;

pos = 2 * (ind - 1) / (param.origAngRes - 1) - 1;