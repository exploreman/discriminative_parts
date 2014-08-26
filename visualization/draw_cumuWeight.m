function weightMap = draw_cumuWeight(database, Dictionary, Regions, vSelectFeat, norms, thresh)


    load(database.HoGfeatPath, 'Feats_HoG', 'grids_x', 'grids_y', 'filename'); 
    Feats = Feats_HoG;

im_feat.feat{1} = Feats;
[maxPosi, maxResp, re] = detect_maxResp_filters_multiPooling_para(Dictionary, Regions, im_feat, vSelectFeat, thresh);

n_parts = size(re, 2);
filename = database.filePath; %['/scratch/jisun/Database/caltech101/data/', fname(65:end-4)];    %
imc = imread(filename);
[r, c, d] = size(imc);
if (d == 3) im = rgb2gray(imc); end
count = 1;
   
weightMap = zeros(r,c);
figure
start = 0; count_map = zeros(r,c);
for l = 1: n_parts %n_img
       
        ws = 8;
        pos = maxPosi(:, l);
        x = min([pos(1) * 8 - 4, (pos(1) + ws - 1) * 8 + 4], r);
        y = min([pos(2) * 8 - 4, (pos(2) + ws - 1) * 8 + 4], c);
        
        %[r_cur,c_cur] = size(re{l});
            
          %if codeSets(l) > 0 % & label(l, 1) ~= 2            
              Resp = max(re{l} + thresh(l), 0); [valid_x, valid_y] = find(Resp > 0);

                for k = 1 : length(valid_x)
                    pos(1) = valid_x(k); pos(2) = valid_y(k);
                    x = round(min([pos(1) * 8 - 4, (pos(1) + ws - 1) * 8 + 4]/ 1, r)) ;
                    y = round(min([pos(2) * 8 - 4, (pos(2) + ws - 1) * 8 + 4] / 1, c));
                    weightMap(x(1) : x(2), y(1) : y(2)) = weightMap(x(1) : x(2), y(1) : y(2)) +  norms(l) * sqrt(Resp(pos(1), pos(2))); %max(codeSets(l) + thresh(l), 0)
                    %weightMap(1 : r_cur * 8, 1 : c_cur * 8) = weightMap(1 : r_cur * 8, 1 : c_cur * 8) + imresize(max(re{l} + thresh(l), 0), 8, 'nearest') * norms(l);%(codeSets(l)) * norms(l);
                end
                %count_map(x(1) : x(2), y(1) : y(2)) = count_map(x(1) : x(2), y(1) : y(2)) + 1;
         % end

            
            
            %weightMap(x(1) : x(2), y(1) : y(2)) = weightMap(x(1) : x(2), y(1) : y(2)) + max(re{l} + thresh(l), 0) * norms(l); 
            %weightMap(1 : r_cur * 8, 1 : c_cur * 8) = weightMap(1 : r_cur * 8, 1 : c_cur * 8) + imresize(max(re{l} + thresh(l), 0), 8, 'nearest');%(codeSets(l)) * norms(l);
            %count_map(x(1) : x(2), y(1) : y(2)) = count_map(x(1) : x(2), y(1) : y(2)) + 1;
 
        if l < 70
        subplot(8, 9, count), imshow(imc);
        rectangle('position',[y(1) x(1) ws * 8 + 1 ws * 8 + 1  ], 'EdgeColor', 'r')
        %title(num2str(codeSets(l)));
        count = count + 1;
        end
        
 
    
end
%valid = find(count_map > 0);
weightMap = weightMap./ max(weightMap(:)) ;
%subplot(8, 9, 71) 
%weightMap = (repmat(weightMap ./ max(weightMap(:)), [1,1,d])> 2e-1) .*double(imc); %weightMap / max(weightMap(:)); imshow(weightMap);
%subplot(8, 9, 72), imshow(uint8(weightMap)); colormap(gray);

figure, subplot(1,2,1); imshow(uint8(weightMap));


diff_10 = sqrt(sum((imc([2:end, end], [1 : end], :) - imc).^2, 3));
edgeWeight_10 = exp(-diff_10 / mean(diff_10(:)))';

diff_11 = sqrt(sum((imc([2:end, end], [2 : end, end], :) - imc).^2, 3));
edgeWeight_11 = exp(-diff_11 / mean(diff_11(:)))';

diff_01 = sqrt(sum((imc([1 : end], [2 : end, end], :) - imc).^2, 3));
edgeWeight_01 = exp(-diff_01 / mean(diff_11(:)))';

diff_m10 = sqrt(sum((imc([1 : end], [2 : end, end], :) - imc).^2, 3));
edgeWeight_m10 = exp(-diff_m10 / mean(diff_m10(:)))';

edgeWeights = [edgeWeight_10(:), edgeWeight_01(:), edgeWeight_11(:), edgeWeight_m10(:)];

%% call graph cut to perform image segmentation
lambda_regu = 2; threshold = 0.2;
seg_mask = gcut_interface(weightMap - threshold, edgeWeights, lambda_regu);
cb_factor = 0.6;
seg_mask_orig = seg_mask;
%% update the results by grab-cut
maxIter = 4; lambda_regu = 6; 
for k = 1 : maxIter    
    theta_0 = []; theta_1 = [];
    % compute color distribution of foreground and background
    fprintf('*** ==Computing likelihood for background colors\n');
    [theta_0 c0_lh] = colorgsm(double(imc),1- seg_mask,6,theta_0);
    fprintf('*** ==Computing likelihood for foreground colors\n');
    [theta_1 c1_lh] = colorgsm(double(imc),seg_mask,6,theta_1);
    
    % update likelihood
    k0_lh =  (c1_lh-c0_lh) * cb_factor + (weightMap - threshold);
    
    % re-compute the segmentation mask
    seg_mask = gcut_interface(k0_lh, edgeWeights, lambda_regu);
end


%% eveluate the 
seg_mask_grabcut = seg_mask;

% show results
seg_mask(find(seg_mask == 0)) = 0.5;
mask = repmat(seg_mask, [1, 1, d]);
bg = zeros(r,c,d);
bg(:,:,1) = 255;
ims = mask.* double(imc) + (1 - mask) .* bg;
subplot(1,2,2); imshow(uint8(ims));
