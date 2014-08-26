function [thetas, Filters,vThresh, L, cost_log] = optimize_regressor_softmax_multicls(database_posi, FTrain, model, params, optiParas, L, stepsize)

% retrieve parameters
num_rand = optiParas.num_posi_rand;
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
lambdas = params.lambdas;
partFeat_dim = params.partFeat_dim;

Filters_ms = model.filters;
vThetas = model.regressor;
vThresh = model.thresholds;

% set up group structure
%partFeat_dim = partFeat_dim.^2 * featDims;
b = 0;
[vNParts, vDims, Filters, groups, vlambda_feat] = Filters2GroupVectors(Filters_ms, b, partFeat_dim, params.lambdas);
num_parts = sum(vNParts);

% optimize softmax regressor by stochastic FISTA

beta = 1.5;
t_k = 1 ; 
t_km1 = 1 ;
xk = [vThetas];
xkm1 = xk;
nIter = 0;
errTolerance = 4e-4; err = 1e+10; L_max = 1e+5;
cost_log = [];cost_data = []; 


% set up group structure for the regressor
num_cls = size(vThetas, 2);
groups = [1:size(vThetas, 1)]' * ones(1, num_cls); 
maxIters = 25;
while (nIter < maxIters & err > errTolerance & L < L_max)  
    
    fprintf(' \n Optimize by stochastic FISTA: iter = %d  \n', nIter);
    
    nIter = nIter + 1 ;
    wk = xk + ((t_km1-1)/t_k)*(xk-xkm1) ;
    
    % randomly select the training data
    fprintf('\n Prepare positive examples  \n');
    %if nIter == 1
    [feats, labels] = randSample(FTrain, params, num_rand, vNParts);
    %end
            
    % compute stochastic gradient: softmax logistic regression
    % w, X, L, num_rand, partFeat_dim, vnum_parts
    [g_filters, g_threshs, g_thetas, f] = CompGradient_Softmax(wk, Filters, vThresh, feats, labels,num_rand,  partFeat_dim, vNParts);  %w, X_posi, X_nega, num_posi,y, num_nega, vDims, num_parts
    
    % Step 2.3: searching Lipschitz parameters
    stop_backtrack = 0;
    gk = zeros(size(g_thetas));
    while ~stop_backtrack        
        gk = wk - (1/L)*g_thetas;     
      
        % soft thresholding for the regression parameters  
        xkp1 = SoftThresholdingGroup_multiFeats(double(gk(:)),groups(:), repmat(vlambda_feat'/L, num_cls, 1),  max(groups(:))); %_Group    
        xkp1 = reshape(xkp1, [], num_cls);
        
        cost_wkp1 = CompCost_Softmax(xkp1, Filters,vThresh,  feats, labels,num_rand, partFeat_dim, vNParts); %CompCost_SSVMLoss_withThresholds_multiFeat_ms
        
        cost2 = f + (xkp1(:)-wk(:))'*g_thetas(:) + (L/2)*norm(xkp1-wk)^2 ;
        
        if (cost_wkp1 <= cost2 || L > L_max)
            stop_backtrack = 1;
        else
            L = L*beta             
        end        
    end
    
    % update filter bank and thresholds
    Filters(1:end-1) =Filters(1:end-1) - stepsize * g_filters;    
    vThresh = vThresh - stepsize * g_threshs;
    
    ind = (groups~=0);
    cost = cost_wkp1; % + sum(vlambda_feat' .* sqrt(accumarray(groups(ind),xkp1(ind).^2))) ; %+ lambda_thre * sum(xkp1(end - num_parts + 1 : end).^2);
    cost_log = [cost_log, cost];
    cost_data = [cost_data, cost_wkp1];
    
    % update parameter using proximal operator
    t_kp1 = 0.5*(1+sqrt(1+4*t_k*t_k)) ;
   
    % compute error
    err = norm(xkp1 - xk);
    
    t_km1 = t_k ;
    t_k = t_kp1 ;
    xkm1 = xk ;
    xk = xkp1 ;
    
    if (rem(nIter, 5) == 0)
       cost_log
       f
    end
end    

thetas = xk;
[Filters] = FiltersVectors2Filters(vNParts, Filters, PartSize_fea);