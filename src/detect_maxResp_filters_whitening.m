% This function is to detect the largest response in an image (spatial
% pyramid), given a set of part filters
function [maxResp_Posi, maxResp, maxFeatures] = detect_maxResp_filters_whitening(Filters_ms, img, params, params_whitening)
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
try
    load(img.featPath, 'Feats', 'Pos', 'Levels');
    Feats;
catch
    ComputeFeatures_multiScale_withWhitening(img, params, params_whitening);
    load(img.featPath, 'Feats', 'Pos', 'Levels');
end


% set filters 
filters = Filters_ms{1}; % single type in patch size

% compute the filter response
num_filters = size(filters, 1);
%Feats(end, :) = 0;
re = reshape(filters,num_filters,[]) * Feats;       % to do ????    

% compute the maximum-response
maxResp_Posi = zeros(3, num_filters);
[val, order]=max(re, [], 2);
maxResp= val;
maxResp_Posi(2:3, :) = Pos(:, order);
maxResp_Posi(1, :) = order;

maxFeatures = Feats(:, order);

