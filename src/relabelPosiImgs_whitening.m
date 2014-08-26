function FPosi = relabelPosiImgs_whitening(database_posi,Filters_opti, params, params_whitening) % PartSize_fea,featDims) 
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
num_train_posi = length(database_posi);

for lp = 1 : num_train_posi
    fprintf(' %d ', lp);
    %fname = database_posi{lp}.featPath;
  
    [maxResp_Index, maxResp, ~] = detect_maxResp_filters_whitening(Filters_opti ,database_posi{lp}, params, params_whitening); % detect the part with the best latent value 
    FPosi.idx{lp} = lp;
    FPosi.maxResp{lp} = maxResp;
    FPosi.Feats{lp} = maxResp_Index; %             
end   