function re = compNormalizedResp(feat,filters, norms_HoG, Scales)
num_filters = length(filters);
resp = fconvssemono(single(feat), filters, 1, num_filters); % ???? to dofconv_var_dim_pooling_ms 
re = [];
for k = 1 : max(Scales)
    idx = find(Scales == k);
    re = [re, cellfun(@(x, y) x./y, resp(idx), repmat(norms_HoG(k), 1, length(idx)), 'UniformOutput', false)];
end
%re;