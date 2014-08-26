function im_color = drawResponseMapinColor(resp)
resp = resp / max(resp(:));
cm = colormap;
map = 0 : 1/ 64 : 1;
[r,c] = size(resp);
im_color = zeros(r,c,3);
for k= 1 : r
    for l = 1 : c
        idx = min(max(round(resp(k, l) * 64 / 1), 1), 64);
        im_color(k,l,:)=cm(idx, :);
    end
end