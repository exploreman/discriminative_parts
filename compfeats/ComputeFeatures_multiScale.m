function [database] = ComputeFeatures_multiScale(database, params)
sbins = params.sbins;
PartSize_fea = params.PartSize_fea; 
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

% compute image pyramidal
num_scales = length(scales);
num_pSizes = length(PartSize_fea);
for k = 1 : num_scales
   scale = scales(k) ;
   im_curr = imresize(im, scale);
   imSize{k} = size(im_curr);
   
   %compute the dense raw features: HoG or SIFT
   [Feats_HoG{k}, grids_x_hog{k}, grids_y_hog{k}] = HoGFeatures(im_curr, sbins);
   
   % compute the per-patch norm for normalization
   for p = 1 : num_pSizes
     f = ones(PartSize_fea(p));
     Norms_HoG{k}{p} = sqrt(conv2(sum(Feats_HoG{k}.^2, 3), f, 'valid'));
   end
   
end

% save the features
savefile = featpath;
[PATHSTR,NAME,EXT] = fileparts(savefile);
if ~isdir(PATHSTR),
    mkdir(PATHSTR);
end;
save(savefile, 'Feats_HoG', 'Norms_HoG', 'grids_x_hog', 'grids_y_hog'); 