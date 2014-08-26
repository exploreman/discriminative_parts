%clear all; 

addpath('/nas/home2/j/jisun/Dropbox/Revision/featureExtract_elemDiscovery/');
addpath('/nas/home2/j/jisun/Dropbox/Revision/databases/');

skipFeatureComputation = 1;
skipTrainTestSetup = 0; 
skipInitFilters = 1 ;
iswhitening = 0;
clusterID = 2;

% set parameters
params.databaseName = 'scene_categories'; %'MITindoors';%'uiucsports'; %'uiucsports'; %'scene_categories';  %'willowactions';
params.lambdas = 0.005; 
params.num_classes = 102;
params.tr_num = 30;
params.Scales_detection = 2.^([-2 : 0.25 : 1.0]); 
% params.Scales_detection = 2.^([-2 : 0.125 : 1]); 
params.sbins = 8;
params.featDims = [32];

if iswhitening == 0
    params.PartSize = [33, 49, 65];
    params.PartSize_fea = floor(params.PartSize/params.sbins);
    params.numClusters = 330;
else
    params.PartSize = [49];
    params.PartSize_fea = floor(params.PartSize/params.sbins);
    params.numClusters = 1000;
end 

params.poolRespPara.pyramid = 2.^([0, 1, 2]);
params.poolRespPara.nScales = 5;

% step 1: set the dataset
databaseName = params.databaseName;
if strcmp(databaseName, 'scene_categories') == 1
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/data/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; % ['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS_whiten/'];
    ResultFolderPath_new = ['/nfs/scratch/jisun/Database/Results_classification/',databaseName,'/new/'];
    DictFolderPath_new = ['/nfs/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];
    
elseif strcmp(databaseName, 'Caltech101') == 1
    
    databaseName ='Caltech101';  
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/data/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; %['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS/'];      
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];
    DictFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Dictionary/',databaseName,'/new/'];  
    
    params.poolRespPara.pyramid = [1, 2, 3];
    params.poolRespPara.nScales = 4;
elseif strcmp(databaseName, 'uiucsports') == 1
    databaseName = 'event_database';
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; %['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS_whiten/'];
    DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];   
    
    params.tr_num = 70
elseif strcmp(databaseName, 'MITindoors') == 1
    databaseName = 'MIT_indoor67';
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/Images/'];   
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/']; % create folders on sequoia 2
    featureFolderPath = featureFolderPath_lc;
    DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];   
    params.poolRespPara.pyramid = [1, 2];
    params.poolRespPara.nScales = 3;
    params.featDims = [33];
    %iswhitening = 1;
    %if iswhitening == 1
    %     featureFolderPath = ['/sequoia/data2/jisun/hog_MITindoor_whiten/']; %featureFolderPath_lc;
    %end
end 



if ~isdir(ResultFolderPath_new),
  mkdir(ResultFolderPath_new);
end;

if ~isdir(DictFolderPath_new),
  mkdir(DictFolderPath_new);
end;

if ~isdir(featureFolderPath_lc),
  mkdir(featureFolderPath_lc);
end;


% Step 2: load the dataset, Compute and save the features of the dataset
dbfolderList = dir(databaseFolderPath) ;
saveDatabase = ['./databases/', 'database_',databaseName, '.mat'];
load('trainTestSplit.mat');  
num_class = length(tr_idx);  
if ~(skipFeatureComputation == 1)
    clabel = 1;
   
    count = 1;
    featpath = featureFolderPath;
    
    for k = 1 : length(database_para)    
        filename = database_para{k}.filePath
        [pathstr, name, ext] = fileparts(filename);         
                
        % feature files
        if iswhitening == 0
            savefile = [featpath, list(d).name, '.mat'];              
        else
            savefile =  [featpath, name, '_hogWhiten', '.mat'];%[featpath, list(d).name, '_whiten.mat']; %
        end
        
        database.featPath{k} = savefile;
        database_para{k}.featPath = savefile;
    end
    
    
    if iswhitening == 0
        if strcmp(databaseName, 'MIT_indoor67') == 1
            % load pre-defined train/test split
            load('trainTestSplit.mat');            
            tr_idx_all = [];
            ts_idx_all = [];
            num_train = length(cell2mat(tr_idx));
            for k = 1 : num_class
                %idx_label = find(Imglabel == k);
                tr_num = length(tr_idx{k});
                tr_idx_all = [tr_idx_all, tr_idx{k}];
                ts_idx_all = [ts_idx_all, ts_idx{k} + num_train];
            end
           APT_run('ComputeFeatures_multiScale_withflip', database_para([tr_idx_all, ts_idx_all]), params, 'UseCluster', 1, 'ClusterID', 2, 'KeepTmp', 1, 'NJobs', min(20, num_class));
        else
           APT_run('ComputeFeatures_multiScale', database_para, params, 'UseCluster', 1, 'ClusterID', 2, 'KeepTmp', 1, 'NJobs', min(34, num_class));
        end
    else   
        setParams;
        if(0)
            % compute whitening parameters on the database
            [whitenmat, datamean] = computeWhitenningMatrx(database_para, params);
            save(['whitenMatrix_',databaseName,'.mat'], 'whitenmat', 'datamean');
        else
            % load whitening parameters
            load(['whitenMatrix_',databaseName,'.mat'], 'whitenmat', 'datamean');
        end
        params_whitening.whitenmat = whitenmat;
        params_whitening.datamean = datamean; 
         
        if(1)
            % compute whitened features
            %idxSet = [tr_idx_all, ts_idx_all];
            featureFolderPath_para{1} = featureFolderPath;
            APT_run('ComputeHogFeature_whiten_perimg', database_para, params_whitening,featureFolderPath_para,'UseCluster', 1, 'ClusterID', 2, 'KeepTmp', 1, 'NJobs', 80);
        end
    end
    save(saveDatabase, 'Imglabel', 'database', 'database_para', 'num_class');
else
    load(saveDatabase, 'Imglabel', 'database', 'database_para');
end
num_class = length(tr_idx); 
round = 1;

% Step 3: begin train and test: divide the database by random split
% Step 3.1: train / test split
if ~(skipTrainTestSetup == 1)           
    num_train = length(cell2mat(tr_idx));  
    ts_label = [];
    tr_idx_all = [];
    tr_label_all = [];
    ts_idx_all = [];
    for k = 1 : num_class
        %idx_label = find(Imglabel == k);
        tr_num = length(tr_idx{k});
        ts_num = length(ts_idx{k});
        tr_label{k} = k * ones(1, tr_num);
        ts_label{k} = k * ones(1, ts_num);
        tr_idx_all = [tr_idx_all, tr_idx{k}];
        ts_idx_all = [ts_idx_all, ts_idx{k} + num_train];
        tr_label_all = [tr_label_all, k * ones(1, tr_num)];            
    end

    % extending data by fliping
    for k = 1 : num_class
        % fliped data to enrich training and testing data
        data_posi = database_para(tr_idx_all(find(tr_label_all == k)));
        data_nega = database_para(tr_idx_all(find(tr_label_all ~= k)));
        data_test = database_para(ts_idx{k});
        data_posi_flip = data_posi;
        data_nega_flip = data_nega;
        data_test_flip = data_test;

        database_posi{k} = cell(1, 2 * length(data_posi_flip));
        database_nega{k} = cell(1, 2 * length(data_nega_flip));
        database_test{k} = cell(1, 2 * length(data_test_flip));
        for p = 1 : length(data_posi_flip)
            [f, fn, ex] = fileparts(data_posi_flip{p}.featPath);
            data_posi_flip{p}.featPath = [f,'/', fn, '_flip', ex];
            database_posi{k}(2 * (p - 1) + 1 : 2 * p) = {data_posi{p}, data_posi_flip{p}};
        end
        for p = 1 : length(data_nega_flip)
            [f, fn, ex] = fileparts(data_nega_flip{p}.featPath);
            data_nega_flip{p}.featPath = [f, '/',fn, '_flip', ex];
            database_nega{k}(2 * (p - 1) + 1 : 2 * p) = {data_nega{p}, data_nega_flip{p}};
        end
        for p = 1 : length(data_test_flip)
            [f, fn, ex] = fileparts(data_test_flip{p}.featPath);
            data_test_flip{p}.featPath = [f, '/',fn, '_flip', ex];
            database_test{k}(2 * (p - 1) + 1 : 2 * p) = {data_test{p}, data_test_flip{p}};
        end                
    end

end

% Step 3.2: perform clustering over the patches crop from each class
saveClusterCenters = [DictFolderPath_new,'ClusterCenters_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_nClass', num2str(1) ,'.mat'];
if ~(skipInitFilters == 1)

    fprintf('round %d: perform clustering over the patches crop from each class', round);
    if iswhitening == 0
        C = APT_run('dictionaryInit_kmeans', tr_idx, database, params, 'UseCluster', 1,  'ClusterID', clusterID, 'KeepTmp', 1, 'NJobs', 34, 'Memory', 8000, 'Libs',{'/scratch/jisun/DownloadCodes/vlfeat-0.9.17/bin/glnxa64'} ); %'ClusterID', 1,
    else            
        % to do: a new version using whitened features
        C = APT_run('dictionaryInit_kmeans_whitening', tr_idx, database, params,params_whitening, 'UseCluster', 1,  'ClusterID', clusterID, 'KeepTmp', 1, 'NJobs',num_class , 'Memory', 8000, 'Libs',{'/scratch/jisun/DownloadCodes/vlfeat-0.9.17/bin/glnxa64'} ); %'ClusterID', 1,
    end
    Filters_Initialization = [];

    for k = 1 : num_class
       Filters_Initialization{k} = C{k};
    end        
    save(saveClusterCenters, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test', 'tr_label', 'ts_label');
else
    load(saveClusterCenters, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','tr_label', 'ts_label');
end

% Step 3.2:  learn discriminative part filters by latent SVM
optiParas.num_posi_rand = 6; 
optiParas.num_nega_rand = 6; 
optiParas.num_iter_out = 8; 
optiParas.num_iter_inn = 80;

if iswhitening == 0    
    [filters, vThreshs] = APT_run('train_parts_LSVM_sparsity', database_posi, database_nega, Filters_Initialization, params, optiParas, 'UseCluster', 1,'ClusterID', clusterID,  'KeepTmp', 1, 'NJobs', 67);
else
    % ???? to do: version for whitened features ????
    [filters, vThreshs] = APT_run('train_parts_LSVM_sparsity_whitening', database_posi, database_nega, Filters_Initialization, params,params_whitening, optiParas, 'UseCluster', 1,'ClusterID', clusterID,  'KeepTmp', 1, 'NJobs', num_class);
end
Filters_opti.filters = filters;
Filters_opti.vThreshs = vThreshs;

% save the learned model
modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Regu_',num2str(params.lambdas),'.mat']
save(modelfile, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','Filters_opti', 'tr_label', 'ts_label');  
clear Filters_Initialization Filters_opti;

% Step 3.3 : train classifier and test
fprintf('round %d: Test by learned model', round);
modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Regu_',num2str(params.lambdas),'.mat']
if iswhitening == 0
    [accuracy, accu] = classification_learnedParts_multiScale(modelfile, DictFolderPath_new, params, round); %FileModel, database, Imglabel, round, PoolingType
else
    % ???? to do: version for whitened features
    [accuracy, accu] = classification_learnedParts_multiScale_whitening_flip(modelfile, DictFolderPath_new, params, params_whitening, round); %FileModel, database, Imglabel, round, PoolingType
end

% save accuracy
save('accu.mat', 'accuracy', 'accu');

% save the results
resultfile = [ResultFolderPath_new, 'Results_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Regu',num2str(params.lambdas),'.mat']   
accu_all{round} = accu;
accu_mean{round} = accuracy;  
save(resultfile, 'accu_all', 'accu_mean');

clear C; 
