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
function [Filters_out, vThresholds_out] = train_parts_LSVM_sparsity_whitening(database_posi, database_nega, Filters_Init, params, params_whitening, optiParas)

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
    Filters_Init{l}  = reshape(Filters_Init{l}, size(Filters_Init{l}, 1), []);
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
    
% Step 3: optimization
iter = 0;
while(iter < num_iter_out)
    fprintf('LSVM-Optimization: iter %d \n', iter);
    
    % relabel positive examples: select the highest scoring latent values
    fprintf(' Relabel positive examples  \n');
    FPosi = relabelPosiImgs_whitening(database_posi,Filters_opti, params, params_whitening) ;
   
    % Optimize part model by latent SVM with group sparsity: stochastic gradient descent
    fprintf(' begin inner loop for optimizaiton  \n');
    [Filters_opti, b, vThreshs] = optimize_filters_whitening(database_posi, database_nega, model, FPosi, params,params_whitening, optiParas);
   
    % set model
    model.filters = Filters_opti;
    model.bias = b;
    model.thresholds = vThreshs;
    
    iter = iter + 1;
end

for l = 1 : num_scale
  Filters_out{l} = Filters_opti{l};                         
  vThresholds_out{l} = vThreshs;
end
