function my_testDropout(name, batchSize, numEpochs, alpha_a, alpha_b, dropout, input_do_rate, hidden_do_rate, ...
    Bayesian_do, alpha_lambda_a, alpha_lambda_b, check_lambda)

cd /home/ichi/work/Boltzman/Bayesin' dropout'/DeepLearningDropout/nn/
addpath ../util;

opt = initializeOptions();
opt.alpha = 1;
opt.adaptive_alpha = true;%false;%true;
opt.alpha_a = alpha_a;%3e4;%50;%10;%1;
opt.alpha_b = alpha_b;%1e4;

opt.batchSize = batchSize;
opt.numEpochs = numEpochs;
opt.numTestEpochs = 1e4;%1e2;%
opt.testerror_dropout = [];%'all';%'last';%
 
opt.dropout = dropout;%true;
opt.testerror_dropout = 'last';%[];%'all';%
opt.gaussian = false;
opt.adaptive = false;% this is NOT opt.adaptive_alpha

opt.input_do_rate = input_do_rate;%0.8; % Probability to set the mask 1 (use the variable)
opt.hidden_do_rate = hidden_do_rate;%0.5;% Probability to set the mask 1 (use the variable)
% 'UOR'(uniformly optimized rate dropout), 
% 'UORH'(uniformly optimized rate dropout for hidden layers), 
% 'LOR'(layer-wise optimized rate dropout) 
% 'FOR'(feature-wise optimized rate dropout) 
opt.Bayesian_do = Bayesian_do;%'UOR';%
opt.check_lambda = check_lambda;
% opt.testerror_dropout = [];%'all';%'last';%
opt.adaptive_alpha_lambda = true;
opt.alpha_lambda_a = alpha_lambda_a;%1e-3*opt.alpha_a;
opt.alpha_lambda_b = alpha_lambda_b;%opt.alpha_b;

start_time = tic;
nn = test_nn(opt);
nn.time = toc(start_time);


filename = filename_writer_nn(name, opt);

save(['files/',filename], 'nn','opt');
