function [im_parts, im_parts_all, resp_maps, resp_maps_nothresh] = showparts_overImages(fname, maxPosi, re, codeSets, wSizes, thresh, Scales, part_labels)

filename = fname; %['/scratch/jisun/Database/caltech101/data/', fname(65:end-4)];    %
imc = imread(filename);
[r, c, d] = size(imc); 
if d == 1
    im(:,:,1) = imc;im(:,:,2) = imc;im(:,:,3) = imc;
    imc = im;
end
im = double(imc);
max_size = max(r, c);min_size = min(r,c);
if (max_size > 300)  
  im = min(max(imresize(im, 300 / max_size), 0), 255); 
end        
if (min_size < 100)
  im = min(max(imresize(im, 120 / min_size), 0), 255);  
end    
[r, c, d] = size(im);
im_orig = im;

Idx_parts = [10 : 50];
im_parts_all = im_orig;

for l = Idx_parts
    ws = wSizes(l);
    scale_curr = maxPosi(1, l);
    pos = maxPosi(2:3, l);

    x = max(min(round([pos(1) * 8 - 4, (pos(1) + ws - 1) * 8 + 4] / Scales(scale_curr)), r), 1);
    y = max(min(round([pos(2) * 8 - 4, (pos(2) + ws - 1) * 8 + 4] / Scales(scale_curr)), c), 1);

        
    % show the max-response parts
    if codeSets(l) > 0
        % draw rectangle on detected parts
        x, y,         
        co = [1,0,0]; %colors(l, :);
        im = drawRectangle(im_orig, x, y, 255 * co);
        im_parts_all = drawRectangle(im_parts_all, x, y, 255 * co);
    end
    
    % show the response map
    resp_map = zeros(size(re{3}{l}));
    resp_map_noThresh = zeros(size(re{3}{l}));
 
    if ~isempty(re)
        resp = max(re{l} + thresh(l), 0);   
        resp_rescale = imresize(re{l}, size(re{l}), 'nearest'); 
        resp_rescale_thresh = imresize(resp, size(re{l}), 'nearest'); 

        resp_map_noThresh = resp_map_noThresh + resp_rescale;
        resp_map = resp_map + resp_rescale_thresh;

        [id_posi_x, id_posi_y] = find(resp > 0);    
    end
  
    
    im_tmp = zeros(size(im)); resp_map = resp_map ./ max(resp_map(:)); resp_map_noThresh = resp_map_noThresh / max(resp_map_noThresh(:));%
    im_tmp(:,:,1) =  0;
    im_tmp(:,:,2) =  0;
    im_tmp(:,:,3) =  255;
    resp_maps{l} = min(max(imresize(resp_map, [r,c], 'nearest'), 0), 1); %im_orig .* repmat(resp_map, [1,1,3]) + zeros(size(im_tmp)) * 0.4 .* (1 - repmat(resp_map, [1,1,3]));
    resp_maps_nothresh{l} = min(max(imresize(resp_map_noThresh, [r,c], 'nearest'), 0), 1);
    im_parts{l} = im;
end
%im_parts = im;
%figure, imshow(uint8(im_parts))
%figure, imagesc((resp_map_all))

function im_rect = drawRectangle(im, xs, ys, color)
[r,c, d] = size(im);im_rect = im;
for k = 1 : d
    im_rect(max(xs(1) - 1, 1) : min(xs(1) + 1,r), ys(1) : ys(2), k) = color(k);
    im_rect(max(xs(2) - 1, 1) : min(xs(2) + 1, r), ys(1) : ys(2), k) = color(k);
    im_rect(xs(1) : xs(2), max(ys(1) - 1, 1) : min(ys(1) + 1, c), k) = color(k);
    im_rect(xs(1) : xs(2), max(ys(2) - 1, 1) : min(ys(2) + 1, c), k) = color(k);
end