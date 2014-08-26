function weigt_map = drawWeightMap(Filters, im_feat)

w = size(im_feat.feat, 1);
h = size(im_feat.feat, 2);

% gather filters for computing match quality responses
num_filters = size(Filters, 4);
width_filter = size(Filters, 1);
dim_feat = size(Filters,3);
filters = cell(num_filters, 1);
weight = zeros(num_filters, 1);
for i = 1:num_filters
    f = Filters(:,:,:,i);
    filters{i} = single(f);
    weight(i) = norm(f);
end
maxPosi = zeros(3, num_filters);

% call the convolution code
maxResp = ones(1, num_filters) * -10;
for level = 1 : length(im_feat)
    
    % make sure that the feature map at current level has larger resolution
    % than filters
    if (size(Filters, 1) < size(im_feat.feat{level}, 1)) && (size(Filters, 2) < size(im_feat.feat{level}, 2)) 
        r = fconv_var_dim(single(im_feat.feat{level}), filters, 1, length(filters));
        Resp{level} = r;    
    end
    
    % find the maximal filter response and corresponding filter label at
    % each patch
    %figure, imagesc(r{88});
    Resp_curr = zeros([size(Resp{level}{1}), num_filters]);
    for k = 1 : num_filters
       Resp_curr(:,:,k) = Resp{level}{k} ;        
    end
    [val_map, idx_map] =max(reshape(Resp_curr,[],num_filters), [], 1);
    
    % find the positions of the maximum response across the pyramid for each
    % filter
    for k = 1 : num_filters
         v = val_map(k);
         idx = idx_map(k);
            if (v > maxResp(k))
                [posi_x, posi_y] = ind2sub(size(Resp{level}{1}), idx);
                maxPosi(:, k) = [level, posi_x, posi_y]';
                maxResp(k) = v;
            end        
    end    
    % 
end
weigt_map = zeros(w + width_filter, h + width_filter); 

% crop the feature set
for k = 1 : num_filters
    scale = maxPosi(1, k);
    pos_x = maxPosi(2, k);
    pos_y = maxPosi(3, k);
    if ~(scale ==0 & pos_x == 0 & pos_y == 0)
        weigt_map((pos_x - 1) * 8 + 1 : (pos_x + width_filter - 1) * 8, (pos_y - 1) * 8 + 1: (pos_y + width_filter - 1) * 8, :) = weigt_map((pos_x - 1) * 8 + 1 : (pos_x + width_filter - 1) * 8, (pos_y - 1) * 8 + 1: (pos_y + width_filter - 1) * 8, :) + norm(k);          
    end
end
