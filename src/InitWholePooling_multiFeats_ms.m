function [Filters_opti, Regions, partDim, num_feat] = InitWholePooling_multiFeats_ms(Filters_Init, Region_Grid,PartSize_fea, featDims )
num_featTypes = length(Filters_Init); num_feat = 0;
for t = 1 : num_featTypes
    num_scale = length(Filters_Init{t});  
    if num_scale > 0
        for l = 1 : num_scale
            num_filter = size(Filters_Init{t}{l}, 1);
            part_size = PartSize_fea(l);
            start = 0;
            num_sele = 1; start = 0;
            Filters_opti{t}{l} = Filters_Init{t}{l};       
            for k = 1 : num_filter   
                Regions{t}{l}{k}.w = [1 : Region_Grid(1)];
                Regions{t}{l}{k}.h = [1 : Region_Grid(2)];
            end

            if t <= 2
               partDim{t}{l} = featDims(t);
            else
               partDim{t}{l} = featDims(t) * PartSize_fea(l) ^ 2;
            end

            num_feat = num_feat + num_filter;
        end
    else
        Filters_opti{t} = [];
        Regions{t} = [];
        partDim{t} = [];
    end        
end