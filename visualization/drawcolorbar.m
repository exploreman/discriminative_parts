colorbar = zeros(64 * 4, 10, 3);
[r,c] = size(colorbar);
cm = colormap;
for k = 1 : r
    ind = min(max(round(k / 4), 1), 64);
    for l = 1 : c
        colorbar(r - k+1, l, :) = cm(ind, :)';    
    end
end
figure, imshow(uint8(colorbar * 255))