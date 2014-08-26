function [g_filters, g_threshs,g_thetas, loss] = CompGradient_Softmax(Weis, Filters, vThresh, X, L, num_rand, partFeat_dim, vnum_parts)
% This function is to compute the gradient of soft-max loss w.r.t. the
% regressor coefficients & the filter banks.

num_cls = size(Weis, 2);
num_parts = 0; 
num_scale = length(partFeat_dim);
for k = 1: num_scale  
    num_parts = num_parts + vnum_parts(k); 
end

Filters = Filters(1 : end-1);

% compute response features
start = 0;
start_thresh = 0;
%score = zeros(num_rand, num_cls);
co = 1;
num_scale = length(partFeat_dim);
feats = [];
for l = 1 : num_scale
    dim_part = partFeat_dim(l);

    viol_parts_posi_curr = zeros(num_rand, vnum_parts(l));
    for k = 1 : vnum_parts(l)
        b = (X(:, start + (k-1) * dim_part + 1 : start + k * dim_part)) * Filters(start + (k-1) * dim_part + 1 : start + k * dim_part) + vThresh(k + start_thresh);
        id = find(b <= 0);
        viol_parts_posi_curr(:, k) = (b > 0);
        b_th = max(b, 0);
        %score = score + b_th * Weis(co, :);
        feats = [feats, b_th];
        co = co + 1;      
    end
    
    start = start + vnum_parts(l) * dim_part;
    start_thresh = start_thresh + vnum_parts(l);   
    viol_parts{l} = viol_parts_posi_curr;
end  

% compute the probability values 
score = feats * Weis;
expVals = exp(score);
probs = expVals ./ repmat(sum(expVals, 2), 1, num_cls);
loss = -mean(sum(L .* log(probs), 2));    

% compute the gradients
g_threshs = zeros(size(vThresh));
g_filters = zeros(size(Filters)); 
g_thetas = zeros(size(Weis));

% regressor
for l = 1 : num_cls
    g_thetas(:, l) = -mean(diag(L(:, l) - probs(:, l)) * feats);
end

% for part detectors
start = 0;
co = 1;
for l = 1 : num_scale
    dim_part = partFeat_dim(l);
    viols = viol_parts{l};
    
    for k = 1 : vnum_parts(l)
        for p = 1 : num_rand
            g_filters(start + (k-1) * dim_part + 1 : start + k * dim_part)= g_filters(start + (k-1) * dim_part + 1 : start + k * dim_part)- sum(Weis(co, :) .* (L(p, :) - probs(p, :)) .* viols(p, k)) * (X(p, start + (k-1) * dim_part + 1 : start + k * dim_part))';
            g_threshs(co) = g_threshs(co) - viols(p, k) * sum(Weis(co, :) .* (L(p, :) - probs(p, :)));
        end
        co = co + 1;
    end
    
    start = start + vnum_parts(l) * dim_part;
end  
g_filters = g_filters / num_rand;
g_threshs = g_threshs / num_rand;
