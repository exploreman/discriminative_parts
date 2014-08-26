function dict = genDictionary(Filters_opti, PartSize_fea, num_perscale)
Dictionary = [];
part_label = [];
scales = [];
norms = []; 
Thresholds = [];
num_scales = length(Filters_opti.filters{1});   
num_classes = length(Filters_opti.filters);

for kk = 1 : num_scales  
     start = 0;
     for k = 1 : num_classes  
        ff = reshape(Filters_opti.filters{k}{kk},size(Filters_opti.filters{k}{kk}, 1), [])';
                  
        ff_norm = sqrt(sum(ff.^2)); [tt, yy] = sort(ff_norm, 'descend');
        ids = find(tt >= 5e-2); len_valid = min(length(ids), num_perscale); %35 %%?????100
        idx_valid = yy(ids(1:len_valid));

        if ~isempty(idx_valid)                         
            ff_valid = ff(:, idx_valid)';ff_norm = sqrt(sum(ff_valid.^2, 2));
            thresh = Filters_opti.vThreshs{k}{kk}(idx_valid)./ ff_norm;

            %id_valid2 = (find(abs(thresh) < 0.95 & thresh <= 0));
            id_valid2 = (find(abs(thresh) < 0.8 & thresh <= 0));
        else
            ff_valid = [];
            id_valid2 = [];
        end

           
                
        if ~isempty(id_valid2);
            ff = ff_valid(id_valid2, :) ./ repmat(ff_norm(id_valid2), 1, size(ff,1));
        else
            ff =[];
        end

        if ~isempty(ff)
            Dictionary{kk}(start + 1: start + length(id_valid2), :) = ff; %reshape(ff', PartSize_fea(kk), PartSize_fea(kk), 32, []);

        end

        Thresholds = [Thresholds; thresh(id_valid2)];
        part_label = [part_label; [k * ones(length(id_valid2), 1), kk * ones(length(id_valid2), 1)]];
        scales = [scales; PartSize_fea(kk) * ones(length(id_valid2), 1)];
        norms = [norms; ff_norm(id_valid2)];

        start = start + length(id_valid2); %size(Filters_opti.filters{k}{t}{kk}, 1)
     end
end

dict.thresholds = Thresholds;
dict.part_label = part_label;
dict.scales = scales;
dict.norms = norms;
dict.filters = Dictionary;
