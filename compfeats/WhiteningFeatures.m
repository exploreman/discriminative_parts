function [Feats_all, Pos, Levels] = WhiteningFeatures(Feats_HoG, Pos_HoG, params)
nls = length(Feats_HoG);

% putting features together
Feats_all = [];
Levels = [];
Pos = [];
for l = 1 : nls
    if ~isempty(Feats_HoG{l})
       Feats_all = [Feats_all, Feats_HoG{l}];
       Levels = [Levels, l * ones(1, size(Pos_HoG{l}, 2))];
       Pos = [Pos, Pos_HoG{l}];
    end   
end

Feats_all = Feats_all';

% normalization before whitening 
Feats_all=bsxfun(@rdivide,bsxfun(@minus,Feats_all,mean(Feats_all,2)),max(sqrt(var(Feats_all,1,2).*size(Feats_all,2)),.0000001));

% whitening
Feats_all=bsxfun(@minus,Feats_all,params.datamean') * (params.whitenmat)';    

% normalization again
Feats_all=single(bsxfun(@rdivide,bsxfun(@minus,Feats_all,mean(Feats_all,2)),max(sqrt(var(Feats_all,1,2).*size(Feats_all,2)),.0000001)));

Feats_all = Feats_all';