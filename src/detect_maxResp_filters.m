% This function is to detect the largest response in an image (spatial
% pyramid), given a set of part filters
function [maxResp_Posi, maxResp, featSize, re] = detect_maxResp_filters(Filters_ms, im_feat,Norms_HoG, params)
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

partFeat_Size = params.PartSize_fea;
featDims = params.featDims;

% gather filters 
num_filters = 0;  
featSize = []; 
filterNorms = [];
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

% compute the filter response
num_scale_detection = length(im_feat.feat);
for s = 1 : num_scale_detection
    num_parts = length(filters);
    if rem(s + 1, 2) == 0 % detection at every 2 scales to decrease the computation overload
        if ~isempty(filters)
            r = size(im_feat.feat{s}, 1); 
            c = size(im_feat.feat{s}, 2); 
            if min(r, c) > max(partFeat_Size)
                re{s}(1 : num_parts) = compNormalizedResp(im_feat.feat{s}, filters, Norms_HoG{s}, Scales);           
            else
                re{s}(1 : num_parts) = cell(1, num_parts);
            end
        end
    end
end

% compute the maximum-response
code = zeros(num_scale_detection, num_filters);
maxResp_Posi_ms = zeros(num_scale_detection,2, num_filters);
maxResp_Posi = zeros(3, num_filters);
for s = 1 : num_scale_detection
    % find max in the current scale
    if ~isempty(re{s})
        for k = 1 : num_filters
           if ~isempty(re{s}{k})
                resp_map = re{s}{k};
                [resp, idx] = max(resp_map(:));            
                code(s, k) = resp;        
                [a,b] = ind2sub(size(re{s}{k}), idx); 
                maxResp_Posi_ms(s, 1:2, k) = [a,b]';
           else
                code(s, k) = 0;
                maxResp_Posi_ms(s, 1:2, k) = [0,0]';
           end
        end
    else
        maxResp_Posi_ms(s, 1:2, 1:num_filters) = repmat([0,0]', 1, num_filters);
        code(s, 1 : num_filters) = 0;
    end           
end

% ???? to do
[val, order]=max(code, [], 1);
maxResp= val;
for k = 1 : num_filters
    maxResp_Posi(2:3, k) = maxResp_Posi_ms(order(k), :, k);            
end    
maxResp_Posi(1, :) = order;

