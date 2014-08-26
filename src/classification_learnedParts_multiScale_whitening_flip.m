function [accuracy, accu]=classification_learnedParts_multiScale_whitening_flip(FileModel, saveFolder, params, params_whitening, round)
trainSVMonCluster = 1;

% retrive parameters
PartSize_fea = params.PartSize_fea;
featDims = params.featDims;
Scales_detect = params.Scales_detection;
poolPara = params.poolRespPara;

% load the learned part dictionary
load(FileModel, 'Filters_Initialization', 'database_posi', 'database_nega', 'database_test' , 'Filters_opti', 'tr_label', 'ts_label');
num_classes = length(Filters_opti.filters);

% extract the part detectors
num_perscale = 180;
dict = genDictionary(Filters_opti, PartSize_fea, num_perscale);
num_words = length(dict.part_label);   
clear Filters_Initialization  Filters_opti;

% coding training image
fprintf('\n Coding each image given a dictionary over the training data set ....');
skipCoding  = 1;
         
PartSize_fea_para{1} = PartSize_fea;
featDims_para{1} = featDims;

% reshape the training data
vlabels_para = [];
co = 1;
for l = 1 : num_classes
    vlabels_para = [vlabels_para, [tr_label{l}, tr_label{l}]];
    for p = 1 : length(database_posi{l})
        database_para{co} = database_posi{l}{p};
        co = co + 1;
    end
end

if ~(skipCoding == 1)      
    % coding_maxResp_filters_whitening(img, dict, poolPara, params, params_whitening)
    APT_run('coding_maxResp_filters_whitening', database_para, dict, poolPara, params, params_whitening ,'UseCluster', 1,'ClusterID', 2,  'KeepTmp', 1, 'NJobs', 40); %, 
end
    
% learn the classifers
if((trainSVMonCluster == 0))
    model=train(vlabels_para', (codeSet), '-c 1');
    clear codeSets codeSet;
else
    labels_para{1} = vlabels_para';
    data_para{1} = database_para;
    c = 0.1;
    model_para=APT_run('trainCluster',labels_para,data_para,c, 'UseCluster', 1, 'KeepTmp', 1,  'ClusterID', 2, 'NJobs', 1, 'Memory', 52000);
    model = model_para{1};
end
%save(savefile_code_train, 'codeSet', 'model', 'tr_label', 'vlabels_para');
%
% save('tmp_whiten.mat', 'model');

% reshape the training data
vlabels_ts_para = [];
co = 1;
for l = 1 : num_classes
    vlabels_ts_para = [vlabels_ts_para, [ts_label{l}, ts_label{l}]];
    for p = 1 : length(database_test{l})
        database_ts_para{co} = database_test{l}{p};
        co = co + 1;
    end
end
% coding the test images
[labels_est, ~, scores_est] = APT_run('predict_maxResp_filters_whitening', database_ts_para, dict, poolPara, model,vlabels_ts_para,params, params_whitening, 'UseCluster', 1, 'KeepTmp', 1, 'ClusterID', 2, 'NJobs',40); %, 
%nimg = length(vlabels_ts_para);
%dfea = length(codeSets{1});
%codeSet = zeros(nimg,dfea);
%for k = 1 : nimg
%    codeSet(k, :) = (sqrt(codeSets{k}));        
%end
%clear codeSets;

% do prediction
%[lc,a]=predict(vlabels_ts_para', (codeSet), model);
%l(start + 1 : start + len) = lc;


scores_est = cell2mat(scores_est);
num_ts = length(database_ts_para) / 2;
for k = 1 : num_ts
   sc = mean([scores_est(2 * (k - 1) + 1, :); scores_est(2 * k, :)]);
   [v, id] = max(sc);
   labels_pred(k) = id;
end

for k = 1 : num_classes
    idx = find(vlabels_ts_para(1:2:end) == k);
    accu(k) = length(find((labels_pred(idx)) == k)) / length(idx);        
end
accuracy = mean(accu)
save('tmp_result.mat', 'accu', 'accuracy');
savefile = [saveFolder, 'codeSet_p', num2str(PartSize_fea), '_d', num2str(num_words),'_round', num2str(round), '_ms_result_noSpar_V4.mat'];
save(savefile, 'accuracy', 'accu', 'l', 'num_words');

if(0)
    cl = 1;
    id = find(tr_label == cl);
    for k=1 : 10
        [codeSets, maxPosi] = maxRespCoding_learnedPooling_para_multiFeats_multiScale(database_para{vtr_idx(id(k))}, Dictionary_para{1}, Regions_para{1}, Thresholds_para{1},PartSize_fea_para{1}, featDims_para{1}, 0); %,
         weightMap = draw_maxResp_multiScale(database_para{vtr_idx(id(k))}.filePath, maxPosi, codeSets, scales, norms, thresh, part_label, cl, 3, Scales_detect);
     end
end