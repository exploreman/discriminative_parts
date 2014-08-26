function [Filters_curr] = FiltersVectors2Filters(vNParts, wk, PartSize_fea)
start = 1;
num_scale = length(vNParts);  
if num_scale > 0
    for l = 1 : num_scale        
        num_feat_curr = vNParts(l);
        dim = (PartSize_fea(l) .^ 2) * 32; % ?? to do 
        Filters_curr{l} = zeros(num_feat_curr, dim);        
        
        for k = 1 : num_feat_curr
            Filters_curr{l}(k, :) = wk(start : start + dim - 1); 
            start = start + dim;
        end
    end
else
    Filters_curr = [];
end
