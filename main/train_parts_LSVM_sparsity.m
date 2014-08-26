%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is to learn discriminative parts using Latent SVM for each
% category by considering other categories as negative examples
%
% Input:
%     FilePaths_Train_posi: the feature file path for the positive training images
%     FilePaths_Train_nega: the feature file path for the negative training
%                           images
%     Filters_Init: the initialized filters learned by clustering over the
%                   positive examples
%
% Output:
%     Filters_opti: the optimized filters for this category.
%
%
%  Written by Jian SUN, XJTU & INRIA/ENS, Oct 2012
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Filters_out, vThresholds_out] = train_parts_LSVM_sparsity(database_posi, database_nega, Filters_Init, params, optiParas)

% step 1: set parameters
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
num_train_posi = length(database_posi);
num_iter_out = optiParas.num_iter_out;

% step 2: begin training by latent SVM algorithm
% get the feature dimnesion and number of features
num_feat = 0;
num_scale = length(Filters_Init);  
partFeat_dim = [];
for l = 1 : num_scale
    num_filter = size(Filters_Init{l}, 1);
    partFeat_dim(l) = featDims * PartSize_fea(l) ^ 2;
    num_feat = num_feat + num_filter;
end
params.partFeat_dim = partFeat_dim;
Filters_opti = Filters_Init;    
vSelectFeat = ones(num_feat, 1);
num_parts = length(vSelectFeat);
vThreshs = 0 * ones(num_parts, 1);
model.filters = Filters_opti;
model.bias = 0;
model.thresholds = vThreshs;
    
% Step 3: optimization codes
iter = 0;
while(iter < num_iter_out)
    fprintf('LSVM-Optimization: iter %d \n', iter);
    
    % relabel positive examples: select the highest scoring latent value
    fprintf(' Relabel positive examples  \n');
    FPosi = relabelPosiImgs(database_posi,Filters_opti, params) ;
   
    % Optimize part model by latent SVM: stochastic gradient descent
    fprintf(' begin inner loop for optimizaiton  \n');
    [Filters_opti, b, vThreshs] = optimize_filters(database_posi, database_nega, model, FPosi, params, optiParas);
   
    % set model
    model.filters = Filters_opti;
    model.bias = b;
    model.thresholds = vThreshs;
    
    iter = iter + 1;
end

% analyze and sorting the codes
fprintf('Analyzing and Sorting the Parts \n');
maxRespSet = [];
for lp = 1 : num_train_posi
    fprintf(' %d ', lp);
    fname = database_posi{lp}.featPath;
    load(fname);
    im_feat.feat = Feats_HoG; 
    [maxResp_Index, maxResp, ~, ~] = detect_maxResp_filters(Filters_opti, im_feat, Norms_HoG, params); % detect the part with the best latent value    
    maxRespSet = [maxRespSet, maxResp'];
end

% discard parts with no response on the positive images
TotalResp = sum(maxRespSet, 2);
start = 1;

num_scale = length(Filters_opti);  
if num_scale > 0
    for l = 1 : num_scale
      % compute the filter norm and discard the filters removed by
      % group sparsity

      num_filters = size(Filters_opti{l}, 1);
      thresh = vThreshs(start : start + num_filters - 1);
      ff = reshape(Filters_opti{l}, num_filters, [])';
      ff_norm = sqrt(sum(ff.^2));

      % discard parts with no response over the positive examples
      id_valid = find(ff_norm > 1e-3 & TotalResp(start : start + num_filters - 1)' > 0); 
      Filters_out{l} = Filters_opti{l}(id_valid, :);                         
      vThresholds_out{l} = thresh(id_valid);
      start = start + num_filters;          
    end
else
    Filters_out = [];
    vThresholds_out = [];
end
