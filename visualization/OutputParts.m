% setup the database to ouput
params.databaseName = 'Caltech101' %'msrc'%'MIT_indoor67'; % ''scene_categories';  %uiucsports'; %'uiucsports'; %'willowactions';
params.lambdas = 0.005; 
params.num_classes = 102;
params.tr_num = 30;
params.Scales_detection = 2.^([-2 : 0.25 : 1.0]); 
params.sbins = 8;
params.featDims = [32];
params.PartSize_fea = [4, 6, 8];
 

% load the learned part detectors
databaseName = params.databaseName;round=1;iswhitening = 0;
if strcmp(databaseName, 'msrc') == 0
    if strcmp(databaseName, 'scene_categories') == 1
        DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];
    elseif strcmp(databaseName, 'Caltech101') == 1
        DictFolderPath_new = ['/scratch/jisun/MidLevelFeature_Learning/Learned_models/Dictionary/',databaseName,'/new/'];  
    elseif strcmp(databaseName, 'uiucsports') == 1
        DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];	
    elseif strcmp(databaseName, 'MIT_indoor67') == 1
        DictFolderPath_new = ['/scratch/jisun/Database/Dictionary/',databaseName,'/new/'];
    end
    modelfile = [DictFolderPath_new, 'LearnedParts_MSMF_', num2str(params.PartSize_fea), '_iswhtened_', num2str(iswhitening), '_Round', num2str(round) ,'_Regu_',num2str(params.lambdas),'.mat']
    load(modelfile);
    
elseif strcmp(databaseName, 'msrc') == 1
    DictFolderPath_new = ['/nas/home2/j/jisun/Dropbox/Revision/Cosegmentation/mat/'];
    load([DictFolderPath_new, 'LearnedParts_msrc.mat']);
end 
database_para = database_posi;

% pre-process the learned part detectors
max_num_perscale = 35;
dict = genDictionary(Filters_opti, params.PartSize_fea, max_num_perscale);
Thresholds = dict.thresholds;
part_label = dict.part_label;
scales = dict.scales;
norms = dict.norms;
Dictionary = dict.filters;
for l = 1 : length(Dictionary)
   numParts_perscale(l) = [size(Dictionary{l}, 1)];
end

%%% start to compute the data for output

folders = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16];
for p = 1 : length(folders)
    cl = folders(p)
    parts_log = [];             
    weightMaps = [];
    ims = [];
    codeSets_log = [];
    Res = [];
    
    % prepare the dictionary
    ns = length(params.PartSize_fea );
    id = find(part_label(:, 1) == cl); 
    id_sc = part_label(id, 2);
    Thresholds_curr = Thresholds(id);
    Dictionary_curr = cell(1,ns); 
    Scales_curr = scales(id);
    counts = ones(1, ns);
    for kk = 1 : length(id)    
        pos = id(kk);
        if id_sc(kk) > 1
            pos = id(kk) - sum(numParts_perscale(1 : id_sc(kk) - 1));
        end
        si = params.PartSize_fea(id_sc(kk))
        Dictionary_curr{id_sc(kk)}(counts(id_sc(kk)), :, :, :) = reshape(Dictionary{id_sc(kk)}(pos, :), si, si, []);
        counts(id_sc(kk)) = counts(id_sc(kk)) + 1;
    end
    
%    if strcmp(databaseName, 'MIT_indoor67') ~= 1
%        id_img = tr_idx{cl};
%    end

    % compute response map
    co = 1;
    for k=1 : 30
        % draw the part detectors   
        k

        % construct HoG images
        count = 1;
        for ll = 1 : ns
            num_filter = size(Dictionary_curr{ll}, 1);
            for kk = 1 : num_filter
                ff = [];
                ff(:,:,:) = Dictionary_curr{ll}(kk, :, :, :);
                HOG = HOGpicture(ff, 20);
                HOG_im{count} = uint8(HOG * (80/(max(ff(:)))));  
                count = count + 1;                    
            end
        end

        % compute part response
%        if strcmp(databaseName, 'MIT_indoor67') == 1
            img = database_para{cl}{k};
%        else
%            img = database_para{id_img(k)}; 
%        end
        [codeSets, maxPosi, re] = maxRespCoding_learnedPooling(img, Dictionary_curr,Thresholds_curr, params); %,
        
        % draw the response maps and extract parts
        [weightMap, im, im_mask, parts, filters] = draw_maxResp_multiScale(img.filePath, maxPosi, re, codeSets, Scales_curr, Thresholds(id), params.Scales_detection, Dictionary_curr);

        %[im_parts, im_parts_all, resps, resp_nothresh] = showparts_overImages(database_para{id_img(k)}.filePath, maxPosi, re, codeSets, Scales_curr, Thresholds(id), params.Scales_detection, part_label(id, :));


        parts_log{co} = parts;             
        weightMaps{co} = weightMap;
        ims{co} = im;
        codeSets_log{co} = codeSets;
        Res{co} = re;
        
        co = co + 1;
    end

    outputFolder = ['/meleze/data1/jisun/Output/',params.databaseName,'/' ,num2str(cl),'/'];%Faces

    if ~isdir(outputFolder),
      mkdir(outputFolder);
    end;

    outputPartsAndTotalResps(parts_log, filters, weightMaps, ims, outputFolder);  
end