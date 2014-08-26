function [database] = ComputeFeatures_multiScale_withWhitening_withflip(database, params, params_whitening)
sbins = params.sbins;
scales = params.Scales_detection;

% scan the folder
featpath = database.featPath;        
filename = database.filePath;
im = double(imread(filename));
[r, c, d] = size(im); max_size = max(r, c);min_size = min(r,c);

% compute the feature    
if (max_size > 300)  
  im = min(max(imresize(im, 300 / max_size), 0), 255); 
end        
if (min_size < 100)
  im = min(max(imresize(im, 120 / min_size), 0), 255);  
end

for l = 1 : 2
    if l == 2
        im = flipdim(im,  2);        
    end
    % compute image pyramidal
    num_scales = length(scales);
    for k = 1 : num_scales
       scale = scales(k) ;
       im_curr = imresize(im, scale);
       imSize{k} = size(im_curr);

       %compute the dense raw features: HoG or SIFT
       [Feats_HoG{k}, Pos{k}] = HoGFeatures_patches(im_curr, sbins, params.PartSize_fea);   
    end

    % Whitening and normalizing these features
    [Feats, Pos, Levels] = WhiteningFeatures(Feats_HoG, Pos, params_whitening);

    % save the features
    savefile = featpath;
    [PATHSTR,NAME,EXT] = fileparts(savefile);
    if l == 2
        savefile = [PATHSTR,NAME,'_flip',EXT];
    end
    if ~isdir(PATHSTR),
        mkdir(PATHSTR);
    end;

    save(savefile, 'Feats', 'Pos', 'Levels'); 
end