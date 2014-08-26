function model=trainCluster(vlabels_para, database_para, C)
num_imgs = length(database_para);
for k = 1 : num_imgs
    % load feature
    load(database_para{k}.featPath);
    
    if k == 1        
        dfea = length(code);
        
        codeSet = zeros(num_imgs,dfea);
    end
    
    if k < 10
        norm(code(:))
    end
    codeSet(k, :) = code;   
    if(rem(k, 100) == 0)
        k
    end
end

% learn classifier
model=train(vlabels_para, (codeSet), ['-c ', num2str(C)]);
