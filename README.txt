% This project aims to learn discriminative part detector for image recognition. They implement the approaches in paper: 
%     [1] Jian Sun, Jean Ponce. Learning Discriminative Part Detectors for Image classification and Cosegmentation, IEEE Conf. Computer Vision (ICCV), 2013
% 
% Written by Jian SUN (jiansun@mail.xjtu.edu.cn) when working at Inria-willow team

% USAGE:
%     1. Revise the pathes in "setup.m" function to link to the dependent codes
%     2. Revise and run "training_discParts_sparsity_main.m" for learning discriminative part detectors, train and test for image classification using these detectors 
%     (2.1) Please download the database (e.g., 15-scenes) to work with, and save the database in your local computer 
%     (2.2) Setup folders in "step 1" in the function of "training_discParts_sparsity_main". 
%     (2.3) Run this main function. It process as followings: 
%         read database in "step 2";  
%         train / test split in "step 3.1";
%         initialize part detectors for each category by k-means clustering in "step 3.2";
%         learn part detectors for each category in "step 3.3"; 
%         train and test for classification in "step 3.4".
%
% NOTICE: For efficiency, highly recommend to use parallel training for 
%     (1) P1: Feature extraction in "step 1"
%     (2) P2: Part initialization for each class in "step 3.2"
%     (3) P3: Learn part detectors for each class in "step 3.3"
%     (4) P4: in function "classification_learnedParts_multiScale_flip" or
%         "classification_learnedParts_multiScale"
%    
%
%     P1-P3 are the code lines in "training_discParts_sparsity_main.m". The parallelization can be implemented on personal laptop based on
%     matlab parallelization toolbox or on cluster using its parallelization toolbox. 