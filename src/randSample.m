function [feats_train, labels] = randSample(FTrain, params, num_posi_rand, vNParts)
num_cls = length(FTrain);
labels = zeros(num_posi_rand, num_cls);
for ln = 1 : num_posi_rand
    fprintf(' %d ', ln);
    
    idx_lv = max(min(round(rand(1) * num_cls), num_cls), 1);% randomnly select the class
    num_imgs = length(FTrain{idx_lv});
    idx_im =  min(max(round(rand(1) * num_imgs), 1), num_imgs);% randomnly select examples from the class
    
    labels(ln, idx_lv) = 1;
    fname = FTrain{idx_lv}.featPath{idx_im};
    load(fname); 
    im_feat.feat = Feats_HoG;
    
    maxPosi = FTrain{idx_lv}.Feats{idx_im};
    maxFeat = cropFeature_multiScale(maxPosi, im_feat, vNParts, params);         
        
    feats_train(ln, :) = maxFeat(:)'; %(:,69 * 12 + 1:69 * 13);                
end 
feats_train = [feats_train, ones(num_posi_rand, 1)];