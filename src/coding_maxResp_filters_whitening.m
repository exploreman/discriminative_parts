% This function is to detect the largest response in an image (spatial
% pyramid), given a set of part filters
function coding_maxResp_filters_whitening(img, dict, poolPara, params, params_whitening)
%
% Inputs 
%     filters: with size of [Mf, Nf, D] (Mf, Nf are two spatial dimensions, D is the feature dimension)
%     im_feats: the image features [M, N, D] (M, N are two spatial dimensions; M, N, should be larger than Mf, Nf respectively)
%
% Outputs
%     maxPosi: find a set of positions for the input filters in the image
%     Resp: the response map for input filter bank
%
%     Writen by Jian SUN, XJTU & INRIA/ENS, on Sep 28, 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pyramid = poolPara.pyramid;
nlevels = poolPara.nScales;
nlevels_spm = length(pyramid);
num_regions = sum(pyramid.^2);

% gather filters 
filters = dict.filters{1}; % ?? to do
nfs = size(filters, 1);

% compute the filter response
try
    load(img.featPath, 'Feats', 'Pos', 'Levels'); % load image features
    Feats;
catch
    ComputeFeatures_multiScale_withWhitening(img, params, params_whitening);
    load(img.featPath);
end
%Feats(end, :) = 0;
re = reshape(filters,nfs,[]) * Feats;    

% compute pooled response features
num_scale_detection = max(Levels);
scales_all = [1 : num_scale_detection];
inter = num_scale_detection / nlevels; 
vThresh = dict.thresholds;
code = [];
for l = 1 : nlevels
    id_sc = find(scales_all > (l-1) * inter & scales_all <= l * inter);
    scales = scales_all(id_sc);
    beta = zeros(nfs, num_regions);
    
    for m = 1 : length(scales)
        s = scales(m);
        idx_curr = find(Levels == s);
        
        % coding at the current scale using SPM
        if ~isempty(idx_curr)
            pos_curr = Pos(:, idx_curr);
            Resp_curr = re(:, idx_curr)';
       
            r = max(pos_curr(1, :)); 
            c = max(pos_curr(2, :));

            Resp = reshape(Resp_curr,[],nfs);
            Resp_thresh = max(Resp + repmat(vThresh',size(Resp, 1) ,1), 0);
            bId = 0;
            for iter1 = 1:nlevels_spm,
                
                wUnit = r / pyramid(iter1);
                hUnit = c / pyramid(iter1);

                % find to which spatial bin each local descriptor belongs                      
                for j = 1 : pyramid(iter1),
                    for i = 1 : pyramid(iter1),
                    bId = bId + 1;
                    sidxBin = find(pos_curr(1, :) > (i - 1) * wUnit & pos_curr(1, :) <= i * wUnit & pos_curr(2, :) > (j - 1) * hUnit & pos_curr(2, :) <= j * hUnit);
                    if isempty(sidxBin),
                        continue;
                    end      
                    if m > 1
                        beta(:, bId) = max(max(Resp_thresh(sidxBin, :), [], 1)', beta(:, bId));
                    else
                        beta(:, bId) = max(Resp_thresh(sidxBin, :), [], 1);
                    end
                    end
                end    
            end   
        end
    end
    %beta = sqrt(beta);
    no = (norm(beta(:)));
    if no > 0;
        beta = beta / no;
    end
    code = [code; beta(:)]; 
end
code = sqrt(code);
code;
save(img.featPath, 'code', '-append');