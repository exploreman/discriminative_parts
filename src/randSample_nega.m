function feats_nega = randSample_nega(database_nega, Filters, params, num_nega_rand, vNParts)
num_traindata_nega = length(database_nega);
idx_rand = randperm(num_traindata_nega);
    
for ln = 1 : num_nega_rand
    fprintf(' %d ', ln);
    fname = database_nega{idx_rand(ln)}.featPath;
    load(fname); 
    im_feat.feat = Feats_HoG;
    [maxResp_Index, maxResp] = detect_maxResp_filters(Filters, im_feat, Norms_HoG, params); % Filters_opti, Regions,im_feat, vSelectFeat, vThreshs, PartSize_fea,featDims, 0                 
    maxFeat = cropFeature_multiScale(maxResp_Index, im_feat, vNParts, params);        
    feats_nega(ln, :) = reshape(maxFeat, 1, []); %maxResp; %feaDims                
end   

feats_nega = [feats_nega, ones(num_nega_rand, 1)];