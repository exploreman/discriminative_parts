function [feats, grids_x, grids_y] = HoGFeatures(im, sbin)
[r, c, d] = size(im);
num_feat_x = round(r/sbin) - 2;
num_feat_y = round(c/sbin) - 2;
gx = [1 : num_feat_x] * sbin;
gy = [1 : num_feat_y] * sbin;
[grids_x, grids_y] = meshgrid(gx, gy);
if(size(im, 3) == 1)
    im = repmat(im, [1, 1, 3]);
end
feats = features(im, sbin);