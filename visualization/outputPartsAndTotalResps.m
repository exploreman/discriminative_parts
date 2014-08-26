function [im_parts] = outputPartsAndTotalResps(parts_log, filters, weightMaps, ims, outputFolder)

%outputFolder = ['/scratch/jisun/Mypapers/Recognition_PartModel/Recognition_PartModel/Figures/airplanes_initialFilters/'];%Faces
if ~isdir(outputFolder),
  mkdir(outputFolder);
end;

num_parts_show = 7;
id_parts = [1:length(parts_log{1})]; %[55,58,59]; %[31,33,34]; %

num_imgs = length(weightMaps);

% output top detected parts (indexed by id_parts)
for l = 1 : length(id_parts)
    id = id_parts(l);
    count = 1;
    for k = 1 : num_imgs
       if ~isempty(parts_log{k})
           resp = parts_log{k}{id}.resp;

           if resp > 0
                part = parts_log{k}{id}.im; 
                parts_output{l}{count} = part;
                Resps_output{l}(count)= resp;
                count = count + 1;
           end
       end
    end
end

r = 64; c = 64;


for l = 1 : min(length(Resps_output), 180)
    l;
    id = id_parts(l);
    count = 1;
    [a, b] = sort(Resps_output{l}, 'descend');
    im_show = ones(r,(c + 5) * (num_parts_show + 1), 3) * 255;
    
    HOG_im = HOGpicture(filters{id}.w, 8);
    HOG_im = uint8(HOG_im * (80/(max(filters{id}.w(:)))));
    
    im_show(1:r,1:c,1) = imresize(HOG_im, [r,c], 'nearest');
    im_show(1:r,1:c,2) = imresize(HOG_im, [r,c], 'nearest');
    im_show(1:r,1:c,3) = imresize(HOG_im, [r,c], 'nearest');
    
    im_show(1:3, [1:c], 1) = 255; im_show(1:3, [1:c], 2) = 0; im_show(1:3, [1:c], 3) = 0;    
    im_show(end-2:end, [1:c], 1) = 255; im_show(end-2:end, [1:c], 2) = 0; im_show(end-2:end, [1:c], 3) = 0;    
    im_show(1:r, [1:3], 1) = 255; im_show(1:r, [1:3], 2) = 0; im_show(1:r, [1:3], 3) = 0;
    im_show(1:r, [c-2:c], 1) = 255; im_show(1:r, [c-2:c], 2) = 0; im_show(1:r, [c-2:c], 3) = 0;
    
    num_parts_show_curr = min(num_parts_show, length(find(a > 0)));
    for k = 2 : num_parts_show_curr + 1
        im = imresize(parts_output{l}{b(k - 1)}, [r, c]);
        if(size(im, 3) == 1)
            im_tmp = []; im_tmp(:,:,1) = im; im_tmp(:,:,2) = im; im_tmp(:,:,3) = im;
        else
            im_tmp = im;
        end
        im_show(:, (k - 1) * (c + 5) + 1 : (k - 1) * (c + 5) + c, :) = im_tmp;
    end
    im_parts{l} = im_show;   
    imwrite(uint8(im_parts{l}), [outputFolder, num2str(l), '_parts.png']);
end

% output response map
for k = 1 : num_imgs
    respMaps = (weightMaps{k});
    respMaps = respMaps / max(respMaps(:));
    im = ims{k};
    
    
    
    imwrite(uint8(im), [outputFolder, num2str(k), '_im.png']);
    [r,c,d] = size(im);mask = repmat(respMaps, [1,1,3]);
    if d ~= 3
        im = repmat(im, [1,1,3]) ;
    end
    
    im2 = zeros(size(im));
    im2(:,:,1) = 0;
    im2(:,:,2) = 0;
    im2(:,:,3) = 255;
    im_mask = im.*mask + (1-mask).*im2;
    imwrite(uint8( im_mask ), [outputFolder, num2str(k), '_resp.png']);
    
end
