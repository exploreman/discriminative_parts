%%%%%%%%%%%
function [Part_Feat, num_feats, pos_rand_list_x, pos_rand_list_y] = cropPartsFromImages_ms_whitening(Feats, levels, pos, PartSize_fea, num_rand_per_imgs)


nps = size(Feats, 2);
Part_Feat = [];

idxSet = min(max(round(rand(1, num_rand_per_imgs) * nps), 1), nps);
Part_Feat{1} = Feats(:, idxSet); 
pos_rand_list_x = pos(1, idxSet);
pos_rand_list_y = pos(2, idxSet);

num_feats = length(idxSet);
