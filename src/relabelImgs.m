function FTrain = relabelImgs(database_train,Filters_opti, params) % PartSize_fea,featDims) 

%for k = 1 : num_cls
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
num_train_cls = length(database_train);

for lp = 1 : num_train_cls
    fprintf(' %d ', lp);
    fname = database_train{lp}.featPath;
    load(fname); 
    im_feat.feat = Feats_HoG;
    [maxResp_Index, maxResp] = detect_maxResp_filters(Filters_opti, im_feat, Norms_HoG, params);
    FTrain.idx{lp} = lp;
    FTrain.maxResp{lp} = maxResp;
    FTrain.Feats{lp} = maxResp_Index; %    
    FTrain.featPath{lp} = fname;
end   
%end