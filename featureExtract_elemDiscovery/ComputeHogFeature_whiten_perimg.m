function imgs = ComputeHogFeature_whiten_perimg(img, params, HOGFolderPath)
I = double(imread(img.filePath));
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

for k = 1 : 2
    % flip the image
    if k == 2
        I = flipdim(I ,2);         
    end
    
    [path,name,ext] = fileparts(img.filePath);    
    
    pyramid = constructFeaturePyramidForImg(I, params);
    pcs=round(params.patchCanonicalSize/params.sBins)-2;  
    [features, levels, indexes] = unentanglePyramid_my(pyramid, pcs,params);
    Feats = single(features)';
    Levels = levels';
    Pos = indexes';
    Scales = pyramid.scales;

    if k == 2    
        savefile = [HOGFolderPath, name, '_hogWhiten_flip.mat'];        
    else
        savefile = [HOGFolderPath, name, '_hogWhiten.mat'];        
    end
    
    save(savefile, 'Feats', 'Pos', 'Levels');
    
    imgs{k} = img;
    imgs{k}.isflip = k - 1;
    imgs{k}.fileHog_warp = savefile;
end