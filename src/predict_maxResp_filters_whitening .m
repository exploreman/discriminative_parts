function [l,s, scores] = predict_maxResp_filters_whitening(img, dict, poolPara, model, label, params, params_whitening)
[code] = coding_maxResp_filters_whitening_return(img, dict, poolPara,params, params_whitening);
[l,s, scores] = predict(label, sqrt(code)', model);