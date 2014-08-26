%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main function to learn discriminative part detectors and
% perform image classification using these part detectors. These codes
% implement the learning approaches in paper: 
%     [1] Jian Sun, Jean Ponce. Learning Discriminative Part Detectors for Image classification and Cosegmentation, IEEE Conf. Computer Vision (ICCV), 2013
% Written by Jian SUN (jiansun@mail.xjtu.edu.cn) when working at Inria-willow team

% USAGE:
%     (1) Please download the database (e.g., 15-scenes), and save the database in your local computer 
%     (2) Setup folders in "step 1" in the following codes. 
%     (3) Run this main function. It process as followings: 
%         read database in "step 2";  
%         train / test split in "step 3.1";
%         initializing part detectors for each category by k-means clustering in "step 3.2";
%         learn part detectors for each category in "step 3.3"; 
%         train and test for classification in "step 3.4".
%
% NOTICE: For efficiency, highly recommend to use parallel training for 
%     (1) P1: Feature extraction in "step 1"
%     (2) P2: Part initialization for each class in "step 3.2"
%     (3) P3: Learn part detectors for each class in "step 3.3"
%     (4) P4: in function "classification_learnedParts_multiScale_flip" or
%         "classification_learnedParts_multiScale"
%     The parallelization can be implemented on personal laptop based on
%     matlab parallelization toolbox or on cluster using its parallelization tool. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

skipFeatureComputation = 1;
skipTrainTestSetup = 0; 
skipInitFilters = 0;
skipTrainFilters = 0;
%clusterID = 2;

% set parameters
params.databaseName = 'Caltech101'; %'scene_categories';  %'MITindoors';%'uiucsports'; %'uiucsports'; %'willowactions';
params.lambdas = 0.005; 
params.num_classes = 102;
params.Scales_detection = 2.^([-2 : 0.25 : 1.0]); 
params.sbins = 8;
params.featDims = [32];
params.PartSize = [33, 49, 65];
params.PartSize_fea = floor(params.PartSize/params.sbins);
params.numClusters = 330;

params.poolRespPara.pyramid = 2.^([0, 1, 2]);
params.poolRespPara.nScales = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% step 1: set the dataset to load and folders for saving results. Please change these folders for the database you are working with.
%  If you are working with other databases, please write your own codes accordingly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
databaseName = params.databaseName;
if strcmp(databaseName, 'scene_categories') == 1 % if work with "scene_categories"
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/data/']; % folder for saving the training/test data
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/']; % folder to save the extracted HoG features 
    featureFolderPath = featureFolderPath_lc;  
    ResultFolderPath_new = ['/nfs/scratch/jisun/Database/Results_classification/',databaseName,'/new/']; % folder to save the classification results
    DictFolderPath_new = ['/nfs/scratch/jisun/Database/Dictionary/',databaseName,'/new/']; % folder to save the learned dictionary of part detectors
    params.tr_num = 100;
elseif strcmp(databaseName, 'Caltech101') == 1 % if work with "Caltech101"
    databaseName ='Caltech101';  
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/data/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; %['/sequoia/data2/jisun/',databaseName, '/MultiFeatures_MS/'];      
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];
    DictFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Dictionary/',databaseName,'/new/'];  
    
    params.tr_num = 30;
    params.poolRespPara.pyramid = [1, 2, 3];
    params.poolRespPara.nScales = 4;
elseif strcmp(databaseName, 'uiucsports') == 1 % if work with "UIUCsports"
    databaseName = 'event_database';
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/'];
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/'];
    featureFolderPath = featureFolderPath_lc; 
    DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];       
    params.tr_num = 70
elseif strcmp(databaseName, 'MITindoors') == 1 % if work with "MITindoors"
    databaseName = 'MIT_indoor67';
    databaseFolderPath = ['/scratch/jisun/Database/',databaseName,'/Images/'];   
    featureFolderPath_lc = ['/meleze/data1/jisun/Database/Features/',databaseName,'/MultiFeatures_MS/']; % create folders on sequoia 2
    featureFolderPath = featureFolderPath_lc;
    DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    ResultFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Results_classification/',databaseName,'/news/'];   
    
    params.poolRespPara.pyramid = [1, 2];
    params.poolRespPara.nScales = 3;    
    params.tr_num = 30;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 2: load the dataset, compute and save the features of the dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dbfolderList = dir(databaseFolderPath) ;
saveDatabase = ['./databases/', 'database_',databaseName, '.mat'];

if ~(skipFeatureComputation == 1)
    clabel = 1;
    count = 1;
    for k = 3 : length(dbfolderList)    
       %if ~(isdir((dbfolderList(k).name)))
          % scan the sub-folder
          subFolderPath = [databaseFolderPath, dbfolderList(k).name, '/'];
          list = dir(subFolderPath);

          featpath = [featureFolderPath, dbfolderList(k).name, '/' ] ; %      
%          if ~isdir(featpath),
%             mkdir(featpath);
%          end;
          
          for d = 1 : length(list)
            if ~(isdir((list(d).name)))
                d
                filename = [subFolderPath, list(d).name]; 
                [pathstr, name, ext] = fileparts(filename);         
                
                % feature files                
                savefile = [featpath, list(d).name, '.mat'];      
                
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
   
    if strcmp(databaseName, 'MIT_indoor67') == 1
        % load pre-defined train/test split
        load('trainTestSplit.mat');            
        tr_idx_all = [];
        ts_idx_all = [];
        num_train = length(cell2mat(tr_idx));
        for k = 1 : num_class           
            tr_num = length(tr_idx{k});
            tr_idx_all = [tr_idx_all, tr_idx{k}];
            ts_idx_all = [ts_idx_all, ts_idx{k} + num_train];
        end
        
        % compute HoG features for each image: for MIT-indoor, compute
        % HoGs for image and its flipped version        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% P1: The following codes should be parallelized for efficiency
        idSet = [tr_idx_all, ts_idx_all];
        for l = 1 : length(idSet)
           ComputeFeatures_multiScale_withflip(database_para{idSet(l)}, params);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% P1: The following codes should be parallelized for efficiency
        for l = 1 : length(database_para)
            ComputeFeatures_multiScale(database_para{l}, params);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    save(saveDatabase, 'Imglabel', 'database', 'database_para', 'num_class');
else
    load(saveDatabase, 'Imglabel', 'database', 'database_para','num_class');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 3: begin learning part detectors, train and test for classification.
%          Database is randomly splited into train and test set. 
%          For "MIT_indoor", we use the default split in the database.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_rounds = 6;
if strcmp(databaseName, 'MIT_indoor67') == 1
    num_rounds = 1;
end

tr_num = params.tr_num;
for round = 1 : num_rounds     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3.1: train / test split
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~(skipTrainTestSetup == 1)    
       
        if strcmp(databaseName, 'MIT_indoor67') == 1
            load('trainTestSplit.mat'); 
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
                data_test = database_para(ts_idx{k} + num_train);
                data_posi_flip = data_posi;
                data_nega_flip = data_nega;
                data_test_flip = data_test;
                
                database_posi{k} = cell(1, 2 * length(data_posi_flip));
                database_nega{k} = cell(1, 2 * length(data_nega_flip));
                database_test{k} = cell(1, 2 * length(data_test_flip));
                for p = 1 : length(data_posi_flip)
                    [f, fn, ex] = fileparts(data_posi_flip{p}.featPath);
                    data_posi_flip{p}.featPath = [f, fn, '_flip', ex];
                    database_posi{k}(2 * (p - 1) + 1 : 2 * p) = {data_posi{p}, data_posi_flip{p}};
                end
                for p = 1 : length(data_nega_flip)
                    [f, fn, ex] = fileparts(data_nega_flip{p}.featPath);
                    data_nega_flip{p}.featPath = [f, fn, '_flip', ex];
                    database_nega{k}(2 * (p - 1) + 1 : 2 * p) = {data_nega{p}, data_nega_flip{p}};
                end
                for p = 1 : length(data_test_flip)
                    [f, fn, ex] = fileparts(data_test_flip{p}.featPath);
                    data_test_flip{p}.featPath = [f, fn, '_flip', ex];
                    database_test{k}(2 * (p - 1) + 1 : 2 * p) = {data_test{p}, data_test_flip{p}};
                end                
            end
        else
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
            num_train = length(cell2mat(tr_idx));
            for k = 1 : num_class
               database_posi{k} = database_para(tr_idx_all(find(tr_label_all == k)));
               database_nega{k} = database_para(tr_idx_all(find(tr_label_all ~= k))); 
               database_test{k} = database_para(ts_idx{k}); %+ num_train
            end           
            
        end       
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3.2: perform clustering over the patches crop from each class
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    saveClusterCenters = [DictFolderPath_new,'ClusterCenters_', num2str(params.PartSize_fea), '_nClass', num2str(round) ,'.mat'];
    if ~(skipInitFilters == 1)
        
        fprintf('round %d: perform clustering over the patches crop from each class', round);
        Filters_Initialization = [];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% P2: The following codes should be parallelized for efficiency
        for k = 1 : num_class
           Filters_Initialization{k} = dictionaryInit_kmeans(tr_idx{k}, database, params);           
        end        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        save(saveClusterCenters, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test', 'tr_label', 'ts_label');
    else
        load(saveClusterCenters, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','tr_label', 'ts_label');
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3.3:  learn discriminative part filters by latent SVM with group sparsity regularization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('round %d: learn discriminative part filters by latent SVM', round);
    optiParas.num_posi_rand = 6; 
    optiParas.num_nega_rand = 6; 
    optiParas.num_iter_out = 8; 
    optiParas.num_iter_inn = 80;
    modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea),  '_Round', num2str(round) ,'_Regu_',num2str(params.lambdas),'.mat']
       
    if skipTrainFilters == 0
  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% P3: The following codes should be parallelized for efficiency
        for l = 1 : num_class
            [filters{l}, vThreshs{l}] = train_parts_LSVM_sparsity(database_posi{l}, database_nega{l}, Filters_Initialization{l}, params, optiParas);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Filters_opti.filters = filters;
        Filters_opti.vThreshs = vThreshs;

        % save the learned model
        save(modelfile, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','Filters_opti', 'tr_label', 'ts_label');  
        clear Filters_Initialization Filters_opti;
    else
        load(modelfile, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test','Filters_opti', 'tr_label', 'ts_label');  
        clear Filters_Initialization Filters_opti;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Step 3.4 : train classifier and test for classification
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('round %d: Test by learned model', round);
    modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_Round', num2str(round) ,'_Regu_',num2str(params.lambdas),'.mat']
   
    if strcmp(databaseName, 'MIT_indoor67') == 1
        [accuracy, accu] = classification_learnedParts_multiScale_flip(modelfile, DictFolderPath_new, params, round); %FileModel, database, Imglabel, round, PoolingType
    else
        [accuracy, accu] = classification_learnedParts_multiScale(modelfile, DictFolderPath_new, params, round); %FileModel, database, Imglabel, round, PoolingType
    end
    
    
    %% save the results
    resultfile = [ResultFolderPath_new, 'Results_', num2str(params.PartSize_fea),  '_Round', num2str(round) ,'_Regu',num2str(params.lambdas),'.mat']   
    accu_all{round} = accu;
    accu_mean{round} = accuracy;  
    save(resultfile, 'accu_all', 'accu_mean');
     
    clear C; 
end