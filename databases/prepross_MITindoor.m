% pre-process MIT-indoor
idx_gif = [];
for k = 1 : length(database_para)
    if  ~isempty(strfind(database_para{(k)}.filePath,'gif'))
        [im, map] = imread(database_para{(k)}.filePath); 
        if size(map, 2) > 0
            for(i=1:size(map,2))
            channel=map(:,i);
            data{i}=channel(im+1);
            end
            im=cat(3,data{:}) * 255; 
            figure,imshow(uint8(im));
            imwrite(uint8(im), database_para{(k)}.filePath);
            
        else
            imwrite(uint8(im), database_para{(k)}.filePath);
        end
        idx_gif = [idx_gif, k];
        figure,imshow(uint8(im)), title(database_para{(k)}.filePath)
    end
end
