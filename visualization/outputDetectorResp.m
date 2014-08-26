function outputDetectorResp(im_parts, im_parts_all, resps, resp_nothresh, filters)
colormap('default')
outfolder = '/scratch/jisun/Mypapers/Recognition_PartModel/Recognition_PartModel/Figures/';
num_parts = length(im_parts);
for k = 1 : num_parts
   imwrite(uint8(im_parts{k} ), [outfolder, num2str(k), '_part.png']);
   
   res = grs2rgb(resps{k}, colormap);
   imwrite(uint8(res * 255), [outfolder, num2str(k), '_res.png']);
   
   res_no = grs2rgb(imresize(resp_nothresh{k}, size(resps{k}), 'nearest'), colormap);
   imwrite(uint8(res_no * 255), [outfolder, num2str(k), '_res_no.png']);
   
   HOG_im = HOGpicture(filters{k}.w, 20);
   HOG_im = uint8(HOG_im * (100/(max(filters{k}.w(:)))));
   imwrite(uint8(HOG_im), [outfolder, num2str(k), '_filter.png']);
end

imwrite(uint8(im_parts_all), [outfolder, num2str(k), '_all.png']);