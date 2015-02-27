%% This code is used for simple running

% clear
% clc
% close all
% home
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.adaptive_alpha = true;%false;%true;
opt.alpha_a = 3e4;%50;%10;%1;
opt.alpha_b = 1e4;

opt.batchSize = 10;
opt.numEpochs = 201;
opt.numTestEpochs = 1e4;%1e2;%
opt.testerror_dropout = 'all';%'last';%
 
opt.dropout = true;
opt.gaussian = false;
opt.adaptive = false;% this is NOT opt.adaptive_alpha

opt.input_do_rate = 0.8; % Probability to set the mask 1 (use the variable)
opt.hidden_do_rate = 0.5;% Probability to set the mask 1 (use the variable)
% 'UOR'(uniformly optimized rate dropout), 
% 'UORH'(uniformly optimized rate dropout for hidden layers), 
% 'LOR'(layer-wise optimized rate dropout) 
% 'FOR'(feature-wise optimized rate dropout) 


opt.testerror_dropout = 'all';%[];%
tic;
opt.Bayesian_do = [];%'UOR';%
[nn, nn_original] = test_nn(opt);
toc;
figure(1);hold off;plot(nn1.testErrors)
nn1 = nn;

opt.numEpochs = 100;%2001;
tic;
opt.Bayesian_do = 'LOR';%
opt.check_lambda = 'all';
opt.testerror_dropout = [];%'all';%'last';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 1e-3*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt);
toc;
figure(2);hold off;plot(nn.testErrors)
nn20 = nn;

opt.numEpochs = 100;%2001;
tic;
opt.Bayesian_do = 'UORH';%
opt.check_lambda = 'all';
opt.testerror_dropout = [];%'all';%'last';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 1e-3*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt,nn1);
toc;
figure(2);hold off;plot(nn.testErrors)
nn20 = nn;

opt.numEpochs = 2001;
tic;
opt.Bayesian_do = 'UORH';%
opt.testerror_dropout = [];%'all';%'last';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 1e-3*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt,nn1);
toc;
figure(2);hold off;plot(nn.testErrors)
nn2 = nn;

tic;
opt.Bayesian_do = 'UORH';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 3e-7*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt,nn1);
toc;
figure(3);hold off;plot(nn.testErrors)
nn3 = nn;

tic;
opt.Bayesian_do = 'UORH';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 1e-6*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt,nn1);
toc;
figure(4);hold off;plot(nn.testErrors)
nn4 = nn;

tic;
opt.Bayesian_do = 'UORH';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = 3e-6*opt.alpha_a;
opt.alpha_lambda_b = opt.alpha_b;
nn = test_nn(opt,nn1);
toc;
figure(5);hold off;plot(nn.testErrors)
nn5 = nn;

% tic;
% opt.Bayesian_do = 'UORH';%
% opt.adaptive_alpha_lambda = true;
% opt.alpha_lambda_a = 1e-5*opt.alpha_a;
% opt.alpha_lambda_b = opt.alpha_b;
% nn = test_nn(opt,nn1);
% toc;
% figure(6);hold off;plot(nn.testErrors)
% nn6 = nn;

tic;
opt.Bayesian_do = [];%'UORH';%
nn = test_nn(opt,nn1);
toc;
figure(6);hold off;plot(nn.testErrors)
nn10 = nn;

figure(7);hold off;plot(nn2.testErrors(201:end));
hold on;plot(nn2.testErrors(201:end),'r');
hold on;plot(nn3.testErrors(201:end),'k');
hold on;plot(nn4.testErrors(201:end),'c');
hold on;plot(nn5.testErrors(201:end),'g');
hold on;plot(nn6.testErrors(201:end),'p');

% tic;
% opt.Bayesian_do = 'UORH';%
% opt.testerror_dropout = 'last';%[];%
% nn2 = test_nn(opt,nn1);
% toc;
% figure(3);hold off;plot(nn.testErrors)

if strcmp(opt.testerror, 'all') || strcmp(opt.testerror, 'last')
    testErrors = nn.testErrors;
else
    testErrors = [];
end
if strcmp(opt.testerror_dropout, 'all') || strcmp(opt.testerror_dropout, 'last')
    testErrorsDropout = nn.testErrorsDropout;
else
    testErrorsDropout = [];
end
if strcmp(opt.trainingerror, 'all') || strcmp(opt.trainingerror, 'last')
    trainingErrors = nn.trainingErrors;
else
    trainingErrors = [];
end

disp(sprintf('alpha: %d batchSize: %d numEpochs: %d error: %d',...
opt.alpha, opt.batchSize, opt.numEpochs, errors_d1(opt.numEpochs)));
