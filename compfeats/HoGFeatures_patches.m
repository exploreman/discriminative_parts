function [feats, pos] = HoGFeatures_patches(im, sbin, pSize)
[r, c, d] = size(im);
num_feat_x = round(r/sbin) - 2;
num_feat_y = round(c/sbin) - 2;
gx = [1 : num_feat_x];
gy = [1 : num_feat_y] ;
[grids_x, grids_y] = meshgrid(gx, gy);
if(size(im, 3) == 1)
    im = repmat(im, [1, 1, 3]);
end
hog_feats = features(im, sbin);

% unfolding the patches
[rh, ch, d] = size(hog_feats);
rv = rh - pSize + 1;
cv = ch - pSize + 1;
feats = [];
pos = [];
co = 1;
if rv > 0 & cv > 0
   for i = 1 : rv
       for j = 1 : cv
         feats(:, co) = reshape(hog_feats(i : i + pSize - 1, j : j + pSize -1, :), [], 1);
         pos(1, co) = i;
         pos(2, co) = j;
         co  = co + 1;
       end
   end
end
