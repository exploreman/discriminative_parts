function FPosi = relabelPosiImgs(database_posi,Filters_opti, params) % PartSize_fea,featDims) 
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
num_train_posi = length(database_posi);

for lp = 1 : num_train_posi
    fprintf(' %d ', lp);
    fname = database_posi{lp}.featPath;
    load(fname); 
    im_feat.feat = Feats_HoG;
    [maxResp_Index, maxResp] = detect_maxResp_filters(Filters_opti, im_feat, Norms_HoG, params);
    FPosi.idx{lp} = lp;
    FPosi.maxResp{lp} = maxResp;
    FPosi.Feats{lp} = maxResp_Index; %             
end   