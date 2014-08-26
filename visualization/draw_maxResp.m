function draw_maxResp(database, maxPosi, codeSets, idxSet, id)
n_img = length(maxPosi);figure
for l = 1 : min(length(idxSet), 25) %n_img
    k = idxSet(l);
    filename = database{k}.filePath;
    im = imread(filename);
    [r, c, d] = size(im);
    if (d == 3) im = rgb2gray(im); end
    pos = maxPosi{k}(:, id);
    x = [pos(1) * 8 - 4, (pos(1) + 9) * 8 + 4];
    y = [pos(2) * 8 - 4, (pos(2) + 9) * 8 + 4];
    
    subplot(5, 5, l), imshow(im);
    rectangle('position',[y(1) x(1) 81 81], 'EdgeColor', 'r')
    title(num2str(codeSets{k}(id)));
end
        
    