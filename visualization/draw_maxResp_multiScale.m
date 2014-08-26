function [weightMap, im, im_mask, parts, filters] = draw_maxResp_multiScale_tmp(fname, maxPosi, re, codeSets, wSizes, thresh, Scales, Dictionary_curr)
filename = fname; 
im = double(imread(filename));
[r, c, d] = size(im);
max_size = max(r, c);min_size = min(r,c);
if (max_size > 300)  
  im = min(max(imresize(im, 300 / max_size), 0), 255); 
end        
if (min_size < 100)
  im = min(max(imresize(im, 120 / min_size), 0), 255);  
end    
[r, c, d] = size(im);
weightMap = zeros(r,c);

count = 1;
for kk = 1 : length(Dictionary_curr)
    for l = 1 : size(Dictionary_curr{kk}, 1);
        filters{count}.w(:,:,:) = Dictionary_curr{kk}(l, :, :,:);
        filters{count}.b = thresh(count);
        count = count + 1;
    end
end
            
id = 1 : length(codeSets);
n_parts = length(id);

for l = 1: n_parts 
    ws = wSizes(l);
    scale_curr = maxPosi(1, id(l));
    pos = maxPosi(2:3, id(l));

    x = max(min(round([pos(1) * 8 - 4, (pos(1) + ws - 1) * 8 + 4] / Scales(scale_curr)), r), 1);
    y = max(min(round([pos(2) * 8 - 4, (pos(2) + ws - 1) * 8 + 4] / Scales(scale_curr)), c), 1);
    if(0)
        if codeSets(l) > 0 % & label(l, 1) ~= 2
            weightMap(x(1) : x(2), y(1) : y(2)) = weightMap(x(1) : x(2), y(1) : y(2)) + codeSets(id(l)) *norms(id(l)) ; %max(codeSets(id(l)) + thresh(id(l)), 0) * norms(id(l)); % 
       end
    end
    if codeSets(l) > 0
        parts{l}.im = im(x(1) : x(end), y(1) : y(end), :);
        parts{l}.resp = codeSets(l);
    else
        parts{l}.resp = 0;
    end

    if l < 64
    %subplot(8, 8, count), imshow(uint8(im));
    %rectangle('position',[y(1) x(1) ws * 8 + 1 ws * 8 + 1  ], 'EdgeColor', 'r')
    %title(num2str(codeSets(id(l))));count = count + 1;
    end
end


num_scale_detection = length(re);
id_parts = [2, 3]
for s =  1 : num_scale_detection
    if ~isempty(re{s});
        for l = 1 : n_parts              
                ws = wSizes(l);
               if codeSets(l) > 0 %  & label(l, 1) ~= 1            

                    Resp = max(re{s}{l} + thresh(l), 0); [valid_x, valid_y] = find(Resp > 0);

                    for k = 1 : length(valid_x)
                        pos(1) = valid_x(k); pos(2) = valid_y(k);
                        x = round(min([pos(1) * 8 - 4, (pos(1) + ws - 1) * 8 + 4]/ Scales(s), r)) ;
                        y = round(min([pos(2) * 8 - 4, (pos(2) + ws - 1) * 8 + 4] / Scales(s), c));
                        weightMap(x(1) : x(2), y(1) : y(2)) = weightMap(x(1) : x(2), y(1) : y(2)) +  Resp(pos(1), pos(2)); %max(codeSets(l) + thresh(l), 0)
                    end
               end
        end
    end

end

weightMap = weightMap / max(weightMap(:)); 
im_mask = repmat(weightMap, [1, 1, d]) .* double(im);
