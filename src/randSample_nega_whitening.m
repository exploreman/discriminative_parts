function feats_nega = randSample_nega_whitening(database_nega, Filters, params, params_whitening,num_nega_rand, vNParts)
num_traindata_nega = length(database_nega);
idx_rand = randperm(num_traindata_nega);
    
for ln = 1 : num_nega_rand
    fprintf(' %d ', ln);
    %fname = database_nega{idx_rand(ln)}.featPath;
    [maxResp_Index, maxResp, maxFeat] = detect_maxResp_filters_whitening(Filters, database_nega{idx_rand(ln)}, params, params_whitening);
         
    feats_nega(ln, :) = reshape(maxFeat, 1, []); %maxResp; %feaDims                
end   

feats_nega = [feats_nega, ones(num_nega_rand, 1)];