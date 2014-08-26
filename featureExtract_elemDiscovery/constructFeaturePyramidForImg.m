function pyramid = constructFeaturePyramidForImg(im, params, levels)
% levels: What level of pyramid to compute the features for.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).
sBins = params.sBins;
'constructfeatpyr';
size(im);
I = im;

canonicalSize=min(size(I(:,:,1)));
canoScale=1;
IS=I;

size(IS)
[rows, cols, chans] = size(IS);
if chans < 3
  I = repmat(I, [1 1 3]);
  fprintf('WARNING: Image has < 3 channels, replicating channels\n');
end

numLevels  = getNumPyramidLevels(rows, cols, params.scaleIntervals, ...
  params.patchCanonicalSize)
scales = getLevelScales(numLevels, params.scaleIntervals);
if nargin < 3 || isempty(levels)
  levels = 1 : numLevels;
end
if(params.useColor == 1)
  im2=RGB2Lab(im).*.0025; 
end

pyramidLevs = cell(1, numLevels);
histbin=-100:20:100;
histbin(end)=histbin(end)+1;
gradientLevs={};
for i = 1 : length(levels)
  lev = levels(i);
  I1 = imresize(I, canoScale / scales(lev),'bilinear');
  [nrows, ncols, unused_dims] = size(I1);
  rowRem = rem(nrows, sBins);
  colRem = rem(ncols, sBins);
  if rowRem > 0 || colRem > 0
    I1 = I1(1:nrows-rowRem, 1:ncols-colRem, :);
  end
  
  feat = features(I1, sBins);
  [rows,cols,~]=size(feat);
  
  feat=feat(:,:,1:31);
  feat=cat(3,feat,imresize(im2(:,:,2),[rows cols],'bilinear'),imresize(im2(:,:,3),[rows cols],'bilinear'));    
  
  [GX, GY] = gradient(I1);
  GI = mean((GX*255).^2, 3) + mean((GY*255).^2, 3);
  GI=imresize(GI,[rows,cols],'bilinear');
  pyramidLevs{lev} = feat;%(:, :, 1:31);
  gradientLevs{lev} = GI;
end
canoSize.nrows = size(im,1);
canoSize.ncols = size(im,2);
pyramid = struct('features', {pyramidLevs}, 'scales', scales, ...
  'canonicalScale', canoScale, 'sbins', sBins, 'canonicalSize', canoSize, 'gradimg', {gradientLevs});
end

function numLev = getNumPyramidLevels(rows, cols, intervals, basePatSize)
lev1 = floor(intervals * log2(rows / basePatSize(1)));
lev2 = floor(intervals * log2(cols / basePatSize(2)));
numLev = min(lev1, lev2) + 1;
end