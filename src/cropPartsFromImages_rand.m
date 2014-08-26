%%%%%%%%%%%
function [Part_Feat, num_feats, pos_rand_list_x, pos_rand_list_y] = cropPartsFromImages_ms(Feats, grids_x, grids_y, PartSize_fea, num_rand_per_imgs)
 
pos_rand_list_x = [];
pos_rand_list_y = [];

num_scales = length(PartSize_fea);
max_PartSize_fea = max(PartSize_fea);
nlvs = length(Feats);
Part_Feat = [];
num_feats = 0;

counts = ones(1, num_scales);
ll = 1;
for k = 1 : num_scales
    ll = 1;
    while ll <= num_rand_per_imgs
         id_rand_scale = max(round(rand(1) * nlvs), 1);
         numFeat_x = size(grids_x{id_rand_scale}, 2);  %x for sift
         numFeat_y = size(grids_x{id_rand_scale}, 1);  %y for sift  

         if (numFeat_x > PartSize_fea(k)) & (numFeat_y > PartSize_fea(k))  
             rand_x_feat_lt = min(max(round(rand(1) * numFeat_x), 1), numFeat_x - PartSize_fea(k));
             rand_y_feat_lt = min(max(round(rand(1) * numFeat_y), 1), numFeat_y - PartSize_fea(k));

             % make sure that the current patch do not have large
             % overlap with previous patches
    %         if ll > 1
    %             iter = 1;
    %             while (min(abs(rand_x_feat_lt - pos_rand_list_x) + abs(rand_y_feat_lt - pos_rand_list_y)) <= 4) && iter < 100
    %                rand_x_feat_lt = min(max(round(rand(1) * numFeat_x), 1), numFeat_x - max_PartSize_fea);
    %                rand_y_feat_lt = min(max(round(rand(1) * numFeat_y), 1), numFeat_y - max_PartSize_fea); 
    %                iter = iter + 1;
    %             end                     
             %end
             pos_rand_list_x = [pos_rand_list_x, rand_x_feat_lt];
             pos_rand_list_y = [pos_rand_list_y, rand_y_feat_lt];

             %part_range_img(:,count) = [rand_x_feat_lt * sbins - round(sbins / 2), (rand_x_feat_lt + PartSize_fea - 1) * sbins + round(sbins / 2), rand_y_feat_lt * sbins - round(sbins / 2) , (rand_y_feat_lt + PartSize_fea - 1) * sbins + round(sbins / 2)];
             %img_path{count} = filename;


             Part_Feat{k}(:,:,:,counts(k)) = Feats{id_rand_scale}(rand_x_feat_lt : rand_x_feat_lt + PartSize_fea(k) - 1, rand_y_feat_lt : rand_y_feat_lt + PartSize_fea(k) - 1, :); 

             %Part_Patch(:,:,:, count) = im(part_range_img(1, count) : part_range_img(2, count), part_range_img(3, count): part_range_img(4, count), :);
             counts(k) = counts(k) + 1;
             ll = ll + 1;
         end
     end
end
try
num_feats = size(Part_Feat{3},4);
catch
   num_feats; 
end