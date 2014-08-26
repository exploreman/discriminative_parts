params_whitening= struct( ...    
        'maxpixels',250000,... % large images will be downsampled to this many pixels.
        'minpixels',200000,... % small images will be upsampled to this many pixels.
        'patchCanonicalSize', {[64 64]}, ... % resolution for each detected patch.
        'scaleIntervals', 6, ... % number of pyramid levels per scale.
        'sBins', 8, ... % pixel width/height for HOG cells
        'useColor', 1, ... % include Lab tiny images in the descriptor for a patch.
        'whitening', 1, ... % whiten the patch features
        'normbeforewhit', 1, ... % mean-subtract and normalize features before applying whitening
        'normalizefeats', 1, ... % mean-subtract and normalize features after applying whitening    
        'samplingNumPerIm',15,... % sample this many patches per image.
        'multperim', 1, ... % allow multiple detections per image
        'nmsOverlapThreshold', 0.4 ... % overlap threshold for NMS during detection.
        );