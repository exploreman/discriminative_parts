function [Filters_curr] = GroupVectors2Filters_whiten(vNParts, wk, PartSize_fea, featdim)
start = 1;
num_scale = length(vNParts);  
if num_scale > 0
    for l = 1 : num_scale
        dim = (PartSize_fea(l) .^ 2) * featdim; % ?? to do         
        num_feat_curr = vNParts(l);
        for k = 1 : num_feat_curr
            Filters_curr{l}(k, :,:,:) = reshape(wk(start : start + dim - 1), PartSize_fea(l), PartSize_fea(l), featdim, []); 
            start = start + dim;
        end
    end
else
    Filters_curr = [];
end
