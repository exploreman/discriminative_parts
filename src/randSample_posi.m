function feats_posi = randSample_posi(database_posi, FPosi, params, num_posi_rand, vNParts)
num_traindata_posi = length(database_posi);
idx_rand = randperm(num_traindata_posi);
    
for ln = 1 : num_posi_rand
    fprintf(' %d ', ln);
    fname = database_posi{idx_rand(ln)}.featPath;
    load(fname); 
    im_feat.feat = Feats_HoG;
    
    maxPosi = FPosi.Feats{idx_rand(ln)};
    maxFeat = cropFeature_multiScale(maxPosi, im_feat, vNParts, params);         
        
    feats_posi(ln, :) = maxFeat(:)'; %(:,69 * 12 + 1:69 * 13);                
end 
feats_posi = [feats_posi, ones(num_posi_rand, 1)];