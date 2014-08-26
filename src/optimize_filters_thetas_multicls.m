function [Filters, vThreshs] = optimize_filters_thetas_multicls(database_posi, model, FPosi, params, optiParas)

% retrieve parameters
num_posi_rand = optiParas.num_posi_rand;
num_nega_rand = optiParas.num_nega_rand;
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
maxIters = optiParas.num_iter_inn;
lambdas = params.lambdas;
partFeat_dim = params.partFeat_dim;

Filters_ms = model.filters;
vThetas = model.thetas;
vThresh = model.thresholds;

% set up group structure
[vNParts, vDims, w0, groups, vlambda_feat] = Filters2GroupVectors(Filters_ms, b, partFeat_dim, params.lambdas);
num_parts = sum(vNParts);

% Step 2: begin stochasitic gradient descent, solve by FISTA with group
% sparsity and stochasitic gradients
L = 40;
beta = 1.5;
t_k = 1 ; 
t_km1 = 1 ;
xk = [double(w0); vThresh];
xkm1 = xk;
nIter = 0;
errTolerance = 4e-4; err = 1e+10; L_max = 1e+5;
cost_log = [];cost_data = []; %figure,
lambda_thre = 0;
labels = [ones([1, num_posi_rand]), -ones([1, num_nega_rand])];
while (nIter < maxIters & err > errTolerance & L < L_max)  
    
    fprintf(' \n Optimize by stochastic FISTA: iter = %d  \n', nIter);
    
    nIter = nIter + 1 ;
    wk = xk + ((t_km1-1)/t_k)*(xk-xkm1) ;
    
    % make filter
    [Filters_curr] = GroupVectors2Filters(vNParts, wk, PartSize_fea);
    
    % randomly select the training data
    fprintf('\n Prepare positive examples  \n');
    feats_posi = randSample_posi(database_posi, FPosi, params, num_posi_rand, vNParts);
            
    % compute stochastic gradient: SSVMLoss
    [f, g] = CompGradient_SSVMLoss(wk, feats_posi, feats_nega, num_posi_rand, labels', num_nega_rand, partFeat_dim, vNParts);  %w, X_posi, X_nega, num_posi,y, num_nega, vDims, num_parts
    
    % Step 2.3: searching Lipschitz parameters
    stop_backtrack = 0;
    gk = zeros(size(g));
    while ~stop_backtrack        
        gk(1:end-num_parts) = wk(1:end-num_parts) - (1/L)*g(1:end-num_parts);     
        grad_g = g(1+end-num_parts:end); 
        
        gk(1+end-num_parts:end) = min( max( wk(1+end-num_parts:end) - (1/L)*grad_g, -1 ), 0);    % (1 + 2 * lambda_thre) * 
        
          
        xkp1_tmp_filters = SoftThresholdingGroup_multiFeats(double(gk(1:end-num_parts )),groups, vlambda_feat/L,  max(groups)); %_Group    
        xkp1_tmp_threshs = gk(end-num_parts + 1 : end) / (1 + 2 * lambda_thre);
        
        xkp1 = [xkp1_tmp_filters; xkp1_tmp_threshs];
        
        cost_wkp1 = CompCost_SSVMLoss(xkp1, feats_posi, feats_nega, num_posi_rand, labels', num_nega_rand, partFeat_dim, vNParts); %CompCost_SSVMLoss_withThresholds_multiFeat_ms
        cost2 = f + (xkp1-wk)'*g + (L/2)*norm(xkp1-wk)^2 ;
        
        if (cost_wkp1 <= cost2 || L > L_max)
            stop_backtrack = 1;
        else
            L = L*beta             
        end        
    end
    
    ind = (groups~=0);
    cost = cost_wkp1 + sum(vlambda_feat' .* sqrt(accumarray(groups(ind),xkp1(ind).^2))) ; %+ lambda_thre * sum(xkp1(end - num_parts + 1 : end).^2);
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
    
    if (rem(nIter, 10) == 0)
       cost_log
       f
    end
end    

Filters = GroupVectors2Filters(vNParts, xk, PartSize_fea);
b = xk(end-num_parts);
vThreshs = xk(end-num_parts + 1 : end); 