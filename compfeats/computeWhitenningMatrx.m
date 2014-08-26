function [whitenmat, datamean]=computeWhitenningMatrx(database, params)
num_imgs = length(database);

ntotal = 0;
sbins = params.sbins;
scales = params.Scales_detection;
for k =1 : 6  : num_imgs   
  k 
  % load image
  im = double(imread(database{k}.filePath));
  [r,c,d] = size(im);
  if d == 1
      im = color(im);
  end
  
  max_size = max(r, c);min_size = min(r,c);

  % compute the feature    
  if (max_size > 300)  
      im = min(max(imresize(im, 300 / max_size), 0), 255); 
  end        
  if (min_size < 100)
      im = min(max(imresize(im, 120 / min_size), 0), 255);  
  end

  % compute image pyramidal
  num_scales = length(scales);
  feats = [];
  Pos = [];
  Levels = [];
  for k = 1 : num_scales
       scale = scales(k) ;
       im_curr = imresize(im, scale);
       imSize{k} = size(im_curr);

       %compute the dense raw features: HoG or SIFT
       [fs, ps] = HoGFeatures_patches(im_curr, sbins, params.PartSize_fea);  
       feats = [feats, fs];
       Pos = [Pos, ps];       
  end
  
  % normalization
  feats = feats';
  feats=bsxfun(@rdivide,bsxfun(@minus,feats,mean(feats,2)),max(sqrt(var(feats,1,2).*size(feats,2)),.0000001));
  feats = feats';   
  
  % sample the image parts
  if(~exist('featsum'))
    featsum=sum(feats,2);
  else
    featsum=featsum+sum(feats,2);
  end
  ntotal=ntotal+size(feats,2);
  if(~exist('dotsum'))
    dotsum=feats*feats';
  else
    dotsum=dotsum+feats*feats';
  end  
end

% compute the whiteninig parameters
covmat=(dotsum./ntotal-(featsum./ntotal)*(featsum'./ntotal));
covmat=covmat+.05*eye(size(covmat,1));
datamean=featsum./ntotal;
disp('performing matrix square root...');
invcovmat=inv(covmat);
whitenmat=sqrtm(invcovmat);