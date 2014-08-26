function Filters = dictionaryInit_kmeans_whitening(tr_idx, database, params, params_whitening) %numClusters, PartSize_fea, num_rand_per_imgs, Scales)
numClusters = params.numClusters;
PartSize_fea = params.PartSize_fea;
Scales = params.Scales_detection;

num_scale = length(PartSize_fea);
num_img = length(tr_idx);
num_feats_sum = 0;


num_parts_all = 1e+5;
num_rand_per_imgs = round(num_parts_all / num_img);
for d = 1 : num_img
   % load features
   
   featurefile = database.featPath{tr_idx(d)};
   try
       load(featurefile);
   catch
       d,
       tr_idx(d),
       img.featPath = database.featPath{tr_idx(d)};
       img.filePath = database.filePath{tr_idx(d)};
       ComputeFeatures_multiScale_withWhitening(img, params, params_whitening);
       load(featurefile);
   end
   %idx = find(levels < ); %find(Scales == 1);
   
   % for HOG feature (directly use the vectors of local hog features as feature of the part)
   [Part_Feat_hog, num_feats_hog, posi_x, posi_y] = cropPartsFromImages_ms_whitening(Feats, Levels, Pos, PartSize_fea, num_rand_per_imgs);
   
   
   for k = 1 : num_scale
        Part_FeatSet{k}(:,num_feats_sum + 1 : num_feats_sum + num_feats_hog) = Part_Feat_hog{k};
   end
   num_feats_sum = num_feats_sum + num_feats_hog;   
   
end

% perform k-means clustering
num_scale = length(Part_FeatSet);
if(num_scale > 0)
    for k = 1 : num_scale 
        Part_Feat_reshape = reshape(Part_FeatSet{k}, [], num_feats_sum);
        Part_Feat_reshape(end, :) = 0;
        %Part_Feat_reshape1 = Part_Feat_reshape ./ repmat(sqrt(sum(Part_Feat_reshape.^2)), size(Part_Feat_reshape, 1), 1);
        %opts = statset('Display','iter');
        %[IDX, C] = kmeans(Part_Feat_reshape1', numClusters, 'EmptyAction', 'drop','maxIter', 150, 'onlinephase', 'off', 'Options',opts); 
        [C, A] = vl_kmeans(Part_Feat_reshape, numClusters, 'algorithm','elkan');
        C = C';
        %norms = sqrt(sum(C.^2, 2)); id = find(norms == 0 | isnan(norms));
        %C(id, :) = [];norms(id) = [];
        Clusters{k} = C; % ./ repmat(norms, 1, size(C, 2));      
    end
else
    Clusters = [];
end

Filters = Clusters;