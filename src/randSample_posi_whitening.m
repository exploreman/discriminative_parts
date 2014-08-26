function feats_posi = randSample_posi_whitening(database_posi, FPosi, params, params_whitening, num_posi_rand, vNParts)
num_traindata_posi = length(database_posi);
idx_rand = randperm(num_traindata_posi);
    
% to do
for ln = 1 : num_posi_rand
    fprintf(' %d ', ln);
    
    try
        fname = database_posi{idx_rand(ln)}.featPath;
        load(fname);
        Feats;
    catch
        ComputeFeatures_multiScale_withWhitening(database_posi{idx_rand(ln)}, params, params_whitening);
        load(database_posi{idx_rand(ln)}.featPath); 
    end
    Feats(end, :) = 0;
    
    maxPosi = FPosi.Feats{idx_rand(ln)};
    maxFeat = Feats(:, maxPosi(1, :));         
        
    feats_posi(ln, :) = maxFeat(:)'; %(:,69 * 12 + 1:69 * 13);                
end 
feats_posi = [feats_posi, ones(num_posi_rand, 1)];