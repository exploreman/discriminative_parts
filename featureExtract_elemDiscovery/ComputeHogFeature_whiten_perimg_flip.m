function [hog_lf, hog_rg] = ComputeHogFeature_whiten_perimg_flip(img, params)
I = double(imread(img));
[r,c,d] = size(I);

if (d == 1)
    I = color(I);
end
% resize the image
if r * c > params.maxpixels
 s = sqrt(params.maxpixels / prod([r,c]));
 I = imresize(I, s);
elseif r * c < params.minpixels
 s = sqrt(params.minpixels / prod([r,c]));
 I = imresize(I, s);
end

% orig
pyramid = constructFeaturePyramidForImg(I, params);
pcs=round(params.patchCanonicalSize/params.sBins)-2;  
[features, levels, indexes, gradsums] = unentanglePyramid(pyramid, pcs,params);
hog_lf.features = features;
hog_lf.levels = levels;
hog_lf.indexes = indexes;
hog_lf.pyramid = pyramid;
hog_lf.gradsums = gradsums;
hog_lf.sim = size(I);

% flip
I2 = flipdim(I ,2); 
pyramid = constructFeaturePyramidForImg(I2, params);
pcs=round(params.patchCanonicalSize/params.sBins)-2;  
[features, levels, indexes, gradsums] = unentanglePyramid(pyramid, pcs,params);
hog_rg.features = features;
hog_rg.levels = levels;
hog_rg.indexes = indexes;
hog_rg.pyramid = pyramid;
hog_rg.gradsums = gradsums;
hog_rg.sim = size(I2);

%[path,name,ext] = fileparts(img.filePath);
%savefile = [HOGFolderPath, name, '_hogWhiten.mat'];
%save(savefile, 'hog', 'pyramid');