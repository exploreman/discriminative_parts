function [codeSet, maxPosi, re] = maxRespCoding_learnedPooling(database, Filters, Thresholds, params)

fname = database.featPath;
load(fname); 
im_feat.feat = Feats_HoG;

% compute the response map
[maxPosi, maxResp, featSize, re] = detect_maxResp_filters(Filters, im_feat, Norms_HoG, params);

beta = maxResp(:);    
codeSet = beta; 
    
beta_thresh = codeSet + Thresholds; %repmat(auto_thresholds,num_words, 1 );
beta_thresh(find(beta_thresh < 0)) = 0;   
isnormalized = 0;
if (isnormalized == 1)
    norm_val = norm(beta_thresh);

    % normalize the code
    if norm_val > 0
        codeSet = beta_thresh / norm_val;       
    else
        codeSet = beta_thresh; 
    end
else
    codeSet = beta_thresh; 
end

codeSet = sparse(codeSet);