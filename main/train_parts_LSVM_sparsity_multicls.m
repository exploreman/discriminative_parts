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
function [Filters_out, vThresholds_out] = train_parts_LSVM_sparsity_multicls(database_posi, Filters_Init, params, optiParas)

% step 1: set parameters
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
num_iter_out = optiParas.num_iter_out;
num_iter_inner = optiParas.num_iter_inn;

% step 2: begin training by latent SVM algorithm
% put the initialized filters together from different classes
num_cls = length(Filters_Init);
num_scale = length(Filters_Init{1}); % number of part sizes
FiltersAll = [];
for l = 1 : num_scale
    FiltersAll{l} = []; 
    for k = 1 : num_cls
       FiltersAll{l} = [FiltersAll{l}; Filters_Init{k}{l}];
    end
end
Filters_Init = FiltersAll;

% get the feature dimnesion and the number of features
num_scale = length(Filters_Init);  
partFeat_dim = [];
num_filter = 0;
for l = 1 : num_scale
    num_filter = num_filter + size(Filters_Init{l}, 1);
    partFeat_dim(l) = featDims * PartSize_fea(l) ^ 2;
    %num_feat = num_feat + num_filter;
end
params.partFeat_dim = partFeat_dim;
Filters_opti = Filters_Init;    
vThreshs = 0 * ones(num_filter, 1);
model.filters = Filters_opti;
model.regressor = randn(num_filter, num_cls) * 1e-2;
model.thresholds = vThreshs;
    
% Step 3: optimization codes
iter = 0; stepsize0 = 1;iter_all = 0; log_model = []; stepsize = stepsize0;
while(iter < num_iter_out)
    fprintf('LSVM-Optimization: iter %d \n', iter);
    
    % relabel all training examples: select the highest scoring latent
    % value (process in parallel)
    fprintf(' Relabel latent variables for all examples  \n');
    Filters_opti_para{1} = Filters_opti;
    FTrain = APT_run('relabelImgs', database_posi,Filters_opti_para, params, 'UseCluster', 1,'ClusterID', 1,  'KeepTmp', 1, 'NJobs', min(51, length(database_posi))) ; % process in parallel
    
    % Optimize part model and softmax logistic regression: stochastic gradient descent
    iter_inner = 1; L = 40; % Lips. parameter
    while iter_inner < num_iter_inner
        fprintf(' begin inner loop for optimizaiton  \n');
       
        % optimize logistic regressor: by stochastic FISTA
        [thetas, Filters_opti, vThreshs, L, cost] = optimize_regressor_softmax_multicls(database_posi, FTrain, model, params, optiParas, L, stepsize);
        
        % optimize part-detectors: stochastic gradient descent
        %[] = optimize_filters_softmax_multicls(database_posi, FTrain, model, params, optiParas);        
        iter_inner = iter_inner + 1;
        iter_all = iter_all + 1;
        
        model.filters = Filters_opti;
        model.regressors = thetas; % to be updated
        model.thresholds = vThreshs;        
        
        stepsize = stepsize0 / sqrt(iter_all);        
        
        if rem(iter_inner, 100) == 0
            iter,
            iter_inner,            
        end
    end
   
    
    % set model
    model.filters = Filters_opti;
    model.regressors = thetas; % to be updated
    model.thresholds = vThreshs;
    
    iter = iter + 1;
    
    log_model{iter} = model;
    save('tmp.mat', 'model', 'iter', 'cost');
end

Filters_out = model.filters;
vThresholds_out = model.thresholds;