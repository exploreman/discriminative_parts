function [vNParts, vDims, w0, groups, vlambda_feat] = Filters2GroupVectors(Filters_ms, bias, partFeat_dim, lambdas)
groups = zeros(sum(partFeat_dim) + 1, 1); 
num_parts = 0;
count = 1;
start = 1;

num_scale = length(Filters_ms);  
if num_scale > 0
    for l = 1 : num_scale
        s = size(Filters_ms{l});
        num_feat_curr = s(1);
        num_parts = num_parts + num_feat_curr;

        dim = partFeat_dim(l);
        vNParts(l) = num_feat_curr;
        vDims(l) = dim * num_feat_curr;

        for k = 1 : num_feat_curr
            groups(start : start + dim - 1) = [count * ones(dim, 1)];   
            w0(start : start + dim - 1) = reshape(Filters_ms{l}(k, :), [], 1);
            start = start + dim; 
            vlambda_feat(count) = lambdas;
            count = count + 1;            
        end 

    end
else
    vNParts{t} = [];
    vDims{t} = [];
end

groups(end + 1) = 0;
w0(end + 1) = bias;
w0 = w0';