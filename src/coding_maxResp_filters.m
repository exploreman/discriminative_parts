% This function is to detect the largest response in an image (spatial
% pyramid), given a set of part filters
function [code] = coding_maxResp_filters(img, dict, partFeat_Size,featDims, poolPara)
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
num_filters = 0;  
featSize = []; 
filterNorms = [];
Filters_ms = dict.filters;
num_scale = length(Filters_ms);
id = 1;
filters = [];
Scales = [];
for k = 1 : num_scale
   num_filters = num_filters + size(Filters_ms{k}, 1);
   for i = 1:size(Filters_ms{k}, 1)
     filters{id} = single(reshape(Filters_ms{k}(i, :), partFeat_Size(k), partFeat_Size(k), featDims , []));
     Scales(id) = k;
     filterNorms = [filterNorms, norm(Filters_ms{k}(i, :))];
     id = id + 1;
   end
end
nfs = length(filters);

% compute the filter response
load(img.featPath);
num_scale_detection = length(Feats_HoG);
for s = 1 : num_scale_detection
    num_parts = length(filters);
     
    if ~isempty(filters)
        r = size(Feats_HoG{s}, 1); 
        c = size(Feats_HoG{s}, 2); 
        if min(r, c) > max(partFeat_Size)
            re{s}(1 : num_parts) = compNormalizedResp(Feats_HoG{s}, filters, Norms_HoG{s}, Scales);           
        else
            re{s}(1 : num_parts) = cell(1, num_parts);
        end
    end
end

% compute pooled response features
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
        
        % coding at the current scale using SPM
        if ~isempty(re{s})
            r = size(re{s}{1}, 1); %ize(idx_map);
            c = size(re{s}{1}, 2);
            Resp_curr = zeros(r, c, num_filters);
            for k = 1 : num_filters
                [r_cur, c_cur] = size(re{s}{k});
                if ~isempty(re{s}{k})
                    Resp_curr(1:r_cur,1:c_cur,k) = re{s}{k};        
                end
            end

            r = size(re{s}{1}, 1); 
            c = size(re{s}{1}, 2);

            Resp = reshape(Resp_curr,[],num_filters);
            Resp_thresh = max(Resp + repmat(vThresh',size(Resp, 1) ,1), 0);
            bId = 0;
            for iter1 = 1:nlevels_spm,

                wUnit = r / pyramid(iter1);
                hUnit = c / pyramid(iter1);

                % find to which spatial bin each local descriptor belongs
                grid_x = repmat([1 : r]', 1,c);
                grid_y = repmat([1 : c], r, 1);
                xBin = ceil(grid_x / wUnit);
                yBin = ceil(grid_y / hUnit);
                idxBin = (yBin - 1)*pyramid(iter1) + xBin;
                
                nBins = pyramid(iter1).^2;        
                for iter2 = 1:nBins,     
                    bId = bId + 1;
                    sidxBin = find(idxBin == iter2);
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
    %beta = sqrt(beta);
    no = (norm(beta(:)));
    if no > 0;
        beta = beta / no;
    end
    code = [code; beta(:)]; 
end

code;