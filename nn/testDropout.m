% clear
% clc
% close all
% home
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.batchSize = 10;
opt.numEpochs = 201;
opt.numTestEpochs=1e4;
opt.testerror_dropout = 'last';
 
opt.dropout = true;
opt.gaussian = false;

opt.adaptive = false;%true;

opt.input_do_rate = 0.5;
opt.hidden_do_rate = 0.5;%opt.input_do_rate;
tic;
[errors_d1, trainingErrors, testErrorsDropout] = test_nn(opt);
toc;
disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d1(opt.numEpochs)));
