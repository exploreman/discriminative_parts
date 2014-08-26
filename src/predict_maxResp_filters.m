function [l,s, vals] = predict_maxResp_filters(img, dict, partFeat_Size,featDims, poolPara, model, label)
[code] = coding_maxResp_filters(img, dict, partFeat_Size,featDims, poolPara);
[l,s, vals] = predict(label, sqrt(code)', model);