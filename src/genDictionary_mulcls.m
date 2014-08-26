function dict = genDictionary_mulcls(Filters_opti, PartSize_fea, num_perscale)
Dictionary = [];
num_scales = length(Filters_opti.filters); 
regressor = Filters_opti.regressor;
import = sum(abs(Filters_opti.regressor'));
start = 1;vThreshs=[];
for kk = 1 : num_scales      
    ffs = Filters_opti.filters{kk};
    len = size(ffs, 1);
    import_curr = import(start : start + len - 1);
    idx = find(import_curr > 0.13);
    
    nos = sqrt(sum(ffs(idx, :)' .^ 2));
    ths = Filters_opti.vThreshs(start : start + len - 1);
    
    Dictionary{kk} =  diag(1./nos) * ffs(idx,:);
    vThreshs = [vThreshs; ths(idx) ./ nos'];
    
    start = start + len;
end

dict.thresholds = vThreshs;
dict.filters = Dictionary;
