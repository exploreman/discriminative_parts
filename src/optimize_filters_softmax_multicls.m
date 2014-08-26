function [Filters, vThreshs] = optimize_filters_softmax_multicls(database_posi, fTrain, model, params, optiParas)

% retrieve parameters
num_rand = optiParas.num_posi_rand; % batch size for stochastic gradient descent
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
maxIters = 10;
partFeat_dim = params.partFeat_dim;

Filters_ms = model.filters;
vThresh = model.thresholds;
vThetas = model.regressor; % the parameters for logistic regressor

% set up group structure: reshape the filter banks into vector with group
% structure
b = 0;
[vNParts, vDims, w0, groups, vlambda_feat] = Filters2GroupVectors(Filters_ms, b, partFeat_dim, params.lambdas);
wk = [double(w0); vThresh]; % to check later
len_filters = length(w0);
len_threshs = length(vThresh);
num_parts = sum(vNParts);

% optimize the filters by stochasitic gradient descent
nIter = 0;
errTolerance = 4e-4; err = 1e+10; L_max = 1e+5;
cost_log = [];cost_data = []; stepsize = 1e-2;
while (nIter < maxIters & err > errTolerance)  
    
    fprintf(' \n Optimize by stochastic FISTA: iter = %d  \n', nIter);
    
    % randomly select the training data
    fprintf('\n Prepare the training examples  \n');
    [featSet, labels] = randSample(fTrain, params, num_rand, vNParts);
            
    % compute stochastic gradient w.r.t. parts ??????
    [g_filters, g_threshs, ~, loss] = CompGradient_Softmax(vThetas, wk(1 : len_filters), wk(len_filters + 1 : end), featSet, labels, num_rand, partFeat_dim, vNParts);  %w, X_posi, X_nega, num_posi,y, num_nega, vDims, num_parts
    
    % stochastic gradient descent: to do ???? decreasing stepsize
    fs = wk(1 : length(g_filters)) - stepsize * g_filters;
    %fs = ProjConstraint(fs);
    ts = wk(length(g_filters) + 2 : end) - stepsize * g_threshs;
    % update parameters
    wk = [fs; 0 ;ts];
    nIter = nIter + 1;
end    

Filters = FiltersVectors2Filters(vNParts, wk, PartSize_fea);
vThreshs = wk(end-num_parts + 1 : end); 

% projection constraint
function fs = ProjConstraint(vNParts, wk, PartSize_fea)
start = 1;
num_scale = length(vNParts);  
if num_scale > 0
    for l = 1 : num_scale
        dim = (PartSize_fea(l) .^ 2) * 32; % ?? to do         
        num_feat_curr = vNParts(l);
        for k = 1 : num_feat_curr
            filter = wk(start : start + dim - 1); %reshape(wk(start : start + dim - 1), PartSize_fea(l), PartSize_fea(l), 32, []); 
            %no = norm(filter);
            %if no > 1
                % projection
            %    filter = filter / no;
            %end
            fs(start : start + dim - 1) = filters(:);
            start = start + dim;
        end
    end
else
    fs = [];
end


