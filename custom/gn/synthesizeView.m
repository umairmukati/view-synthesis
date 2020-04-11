function [Y, R, D] = synthesizeView(ctl, cbl, ctr, cbr, index, mypy)

% Corn shape [372, 540, 4, 3], index shape [2]

if ndims(ctl) == 2
    ctl = repmat(ctl,[1,1,3]);
    cbl = repmat(cbl,[1,1,3]);
    ctr = repmat(ctr,[1,1,3]);
    cbr = repmat(cbr,[1,1,3]);
end

corners = permute(cat(4, ctl, cbl, ctr, cbr),[1 2 4 3]);

out = mypy.synthesizeView(py.torch.tensor(py.numpy.array(corners)).type('torch.FloatTensor'), py.list(uint8(index)));

Y = (single(out{1}.data.cpu().numpy())+1)/2;
R = (single(out{2}.data.cpu().numpy()));
D = (single(out{3}.data.cpu().numpy()));