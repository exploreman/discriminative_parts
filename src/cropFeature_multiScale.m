%%%%%%%%%%%%%%%%%%%%%
%% Crop the HOG features
function Feat = cropFeature_multiScale(maxPosi, im_feat, vNFeat, params)

partFeat_dim = params.partFeat_dim;
partSize = params.PartSize_fea;

start = 1; 
start_posi = 0;
num_scale = length(vNFeat);
    
for l = 1 : num_scale
    num_feat = vNFeat(l);
    dim = partFeat_dim(l);
    
    for k = 1 : num_feat
        %k = idx_seleFeats(p);
        scale_detection = maxPosi(1, start_posi + k);
        pos_x = maxPosi(2, start_posi + k);
        pos_y = maxPosi(3, start_posi + k);
        width_filter = partSize(l);

        if (pos_x ~= 0 & pos_y ~= 0)   
            Fp= im_feat.feat{scale_detection}(pos_x : pos_x + width_filter - 1, pos_y: pos_y + width_filter - 1, :);   
            v = norm(Fp(:));

            if v ~= 0
                Feat(start : start + dim - 1) = Fp(:) / v;
            else
                Feat(start : start + dim - 1) = 0;
            end
        else
            Feat(start : start + dim - 1) = 0;
        end
        start = start + dim;
     end
     start_posi = start_posi + num_feat;
end