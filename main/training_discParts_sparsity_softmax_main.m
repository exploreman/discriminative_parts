%clear all; 

skipFeatureComputation = 1;
skipTrainTestSetup = 0; 
skipInitFilters = 0 ;
iswhitening = 0;
clusterID = 1;

% set parameters
params.databaseName = 'scene_categories'; %'uiucsports'; %  %'willowactions';'Caltech101';
params.lambdas = 2e-4; 
params.num_classes = 15;
params.tr_num = 100;
params.Scales_detection = 2.^([-2 : 0.25 : 1.0]); 
% params.Scales_detection = 2.^([-2 : 0.125 : 1]); 
params.sbins = 8;
params.featDims = [32];
params.tr_num = 100;

if iswhitening == 0
    params.PartSize = [33, 49, 65];
    params.PartSize_fea = floor(params.PartSize/params.sbins);
    params.numClusters = 100;
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
    featureFolderPath = featureFolderPath_lc; %;['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS_whiten/'];
    ResultFolderPath_new = ['/nfs/scratch/jisun/Database/Results_classification/',databaseName,'/new/'];
    DictFolderPath_new = ['/nfs/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];
    
elseif strcmp(databaseName, 'Caltech101') == 1
    
    databaseName ='Caltech101';  
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/data/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; %['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS/'];      
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];
    DictFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Dictionary/',databaseName,'/new/'];  
elseif strcmp(databaseName, 'uiucsports') == 1
    
    databaseName = 'event_database';
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; %['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS_whiten/'];
    DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    ResultFolderPath_new = ['/scratch/jisun/Database/Results_classification/',databaseName,'/new/'];
elseif strcmp(database, 'MITindoors') == 1
    
    
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
num_rand_per_imgs = 150;
dbfolderList = dir(databaseFolderPath) ;
saveDatabase = [featureFolderPath_lc, 'database_',databaseName,'_para_mf_multiScale_isWhitened', num2str(iswhitening), '.mat'];

if ~(skipFeatureComputation == 1)
    clabel = 1;
   
    count = 1;
    for k = 3 : length(dbfolderList)    
       %if ~(isdir((dbfolderList(k).name)))
          % scan the sub-folder
          subFolderPath = [databaseFolderPath, dbfolderList(k).name, '/']
          list = dir(subFolderPath);

          featpath = [featureFolderPath, dbfolderList(k).name, '/'] ;        
%          if ~isdir(featpath),
%             mkdir(featpath);
%          end;
          
          for d = 1 : length(list)
            if ~(isdir((list(d).name)))
                d
                filename = [subFolderPath, list(d).name];             

                % feature files
                if iswhitening == 0
                    savefile = [featpath, list(d).name, '.mat'];              
                else
                    savefile = [featpath, list(d).name, '_whiten.mat']; 
                end

                database.label{count} = clabel;
                database.filePath{count} = filename;
                database.featPath{count} = savefile;
                
                database_para{count}.label = clabel;
                database_para{count}.filePath = filename;
                database_para{count}.featPath = savefile;
                
                Imglabel(count) = clabel;
                count = count + 1;
            end        
          end

          clabel = clabel + 1;

          %end     
    end
    num_class = clabel - 1;  
    if iswhitening == 0
        APT_run('ComputeFeatures_multiScale', database_para, params, 'UseCluster', 1, 'ClusterID', 1, 'KeepTmp', 1, 'NJobs', min(num_class, 51));
    else   
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
            APT_run('ComputeFeatures_multiScale_withWhitening', database_para, params, params_whitening,'UseCluster', 1, 'ClusterID', 2, 'KeepTmp', 1, 'NJobs', min(num_class, 51));
        end
    end
    save(saveDatabase, 'Imglabel', 'database', 'database_para', 'num_class');
else
    load(saveDatabase, 'Imglabel', 'database', 'database_para','num_class');
end

% Step 3: begin train and test: divide the database by random split
num_rounds = 6;
tr_num = params.tr_num;
for round = 1 : num_rounds     
    
    % Step 3.1: train / test split
    if ~(skipTrainTestSetup == 1)    
        tr_idx = [];
        tr_label = [];
        ts_idx = [];
        ts_label = [];
        tr_idx_all = [];
        tr_label_all = [];
        for k = 1 : num_class
            idx_label = find(Imglabel == k);
            num = length(idx_label);

            idx_rand = randperm(num);
            tr_idx{k} = idx_label(idx_rand(1:tr_num));
            tr_label{k} = k * ones(1, tr_num);
            tr_idx_all = [tr_idx_all, tr_idx{k}]; 
            tr_label_all = [tr_label_all, k * ones(1, tr_num)];

            ts_idx{k} = idx_label(idx_rand(tr_num+1:end)); 
            ts_label{k} = k * ones(1, num - tr_num);
        end
        
        for k = 1 : num_class
           database_posi{k} = database_para(tr_idx_all(find(tr_label_all == k)));
           database_nega{k} = database_para(tr_idx_all(find(tr_label_all ~= k))); 
           database_test{k} = database_para(ts_idx{k});
        end
    end
    % Step 3.2: perform clustering over the patches crop from each class
    saveClusterCenters = [DictFolderPath_new,'ClusterCenters_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_nClass', num2str(round) ,'.mat'];
    if ~(skipInitFilters == 1)
        
        fprintf('round %d: perform clustering over the patches crop from each class', round);
        if iswhitening == 0
            C = APT_run('dictionaryInit_kmeans', tr_idx, database, params, 'UseCluster', 1,  'ClusterID', clusterID, 'KeepTmp', 1, 'NJobs', 51, 'Memory', 8000, 'Libs',{'/scratch/jisun/DownloadCodes/vlfeat-0.9.17/bin/glnxa64'} ); %'ClusterID', 1,
        else            
            % to do: a new version using whitened features
            C = APT_run('dictionaryInit_kmeans_whitening', tr_idx, database, params,params_whitening, 'UseCluster', 1,  'ClusterID', clusterID, 'KeepTmp', 1, 'NJobs', min(num_class, 51), 'Memory', 8000, 'Libs',{'/scratch/jisun/DownloadCodes/vlfeat-0.9.17/bin/glnxa64'} ); %'ClusterID', 1,
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
    fprintf('round %d: learn discriminative part filters by latent SVM', round);
    optiParas.num_posi_rand = 3; 
    optiParas.num_nega_rand = 3; 
    optiParas.num_iter_out = 20; 
    optiParas.num_iter_inn = 1000;
    
    if iswhitening == 0    
        [filters, vThreshs] = train_parts_LSVM_sparsity_multicls(database_posi, Filters_Initialization, params, optiParas);
    end
    
    Filters_opti.filters = filters;
    Filters_opti.vThreshs = vThreshs;
    
    % save the learned model
    modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Round', num2str(round) ,'_Regu_',num2str(params.lambdas),'_softmax.mat']
    save(modelfile, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','Filters_opti', 'tr_label', 'ts_label');  
    clear Filters_Initialization Filters_opti;
    
    % Step 3.3 : train classifier and test
    fprintf('round %d: Test by learned model', round);
    modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Round', num2str(round) ,'_Regu_',num2str(params.lambdas),'_softmax.mat']
    [accuracy, accu] = classification_learnedParts_softmax(modelfile, DictFolderPath_new, params, round); %FileModel, database, Imglabel, round, PoolingType
    
    % save accuracy
    save('accu.mat', 'accuracy', 'accu');
    
    % save the results
    resultfile = [ResultFolderPath_new, 'Results_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Round', num2str(round) ,'_Regu',num2str(params.lambdas),'_softmax.mat']   
    accu_all{round} = accu;
    accu_mean{round} = accuracy;  
    save(resultfile, 'accu_all', 'accu_mean');
     
    clear C; 
end