%% This code is used for paraemter search at maui

addpath ../util;

opt = initializeOptions();
opt.gaussian = false;
opt.adaptive = false;% this is NOT opt.adaptive_alpha
opt.testerror_dropout = [];%'all';%
opt.adaptive_alpha_lambda = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
opt.adaptive_alpha = true;%false;%true;
opt.alpha_a = 1e4;%50;%10;%1;
opt.alpha_b = 1e4;
 
opt.dropout = true;
opt.input_do_rate = 0.8; % Probability to set the mask 1 (use the variable)
opt.hidden_do_rate = 0.5;% Probability to set the mask 1 (use the variable)

% 'UOR'(uniformly optimized rate dropout), 
% 'UORH'(uniformly optimized rate dropout for hidden layers), 
% 'LOR'(layer-wise optimized rate dropout) 
% 'FOR'(feature-wise optimized rate dropout) 
opt.check_lambda = 'all';
% opt.alpha_lambda_a = 1e-3*opt.alpha_a;
% opt.alpha_lambda_b = opt.alpha_b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
name = 'nn_gisette';
opt.batchSize = 10;
opt.numEpochs = 2e3;%1e2;%


%%%%%%%%%%%%%%% Without Droput
opt.alpha_lambda_b = 1; 
opt.alpha_lambda_a = 1;
opt.dropout = false;  
command = sprintf('echo ''matlab -nojvm -r "my_testDropout(''\\''''%s''\\'''',%d,%d,%f,%f,%d,%f,%f,''\\''''%s''\\'''',%f,%f,''\\''''%s''\\'''');quit;"''| qsub -lnodes=1:ppn=2', ...
    name,opt.batchSize, opt.numEpochs, opt.alpha_a, opt.alpha_b, opt.dropout, opt.input_do_rate, opt.hidden_do_rate, ...
    opt.Bayesian_do, opt.alpha_lambda_a, opt.alpha_lambda_b, opt.check_lambda);
unix(command);
pause(0.1);

%%%%%%%%%%%%%%% With Droput
opt.dropout = true; 
opt.Bayesian_do = [];%
command = sprintf('echo ''matlab -nojvm -r "my_testDropout(''\\''''%s''\\'''',%d,%d,%f,%f,%d,%f,%f,''\\''''%s''\\'''',%f,%f,''\\''''%s''\\'''');quit;"''| qsub -lnodes=1:ppn=2', ...
    name,opt.batchSize, opt.numEpochs, opt.alpha_a, opt.alpha_b, opt.dropout, opt.input_do_rate, opt.hidden_do_rate, ...
    opt.Bayesian_do, opt.alpha_lambda_a, opt.alpha_lambda_b, opt.check_lambda);
unix(command);
pause(0.1);

%%
for rate = [1e-1 3e-2 1e-2 3e-3]%[1e-1 3e-2 1e-2 3e-3 1e-3]
%     for b = [1e2 1e3 1e4]
%         k=1;
    opt.alpha_lambda_b = opt.alpha_b;
    opt.alpha_lambda_a = opt.alpha_b*rate;

    %%%%%%%%%%%%%%% With Droput
    opt.dropout = true;
    opt.Bayesian_do = 'UORH';%
    command = sprintf('echo ''matlab -nojvm -r "my_testDropout(''\\''''%s''\\'''',%d,%d,%f,%f,%d,%f,%f,''\\''''%s''\\'''',%f,%f,''\\''''%s''\\'''');quit;"''| qsub -lnodes=1:ppn=2', ...
        name,opt.batchSize, opt.numEpochs, opt.alpha_a, opt.alpha_b, opt.dropout, opt.input_do_rate, opt.hidden_do_rate, ...
        opt.Bayesian_do, opt.alpha_lambda_a, opt.alpha_lambda_b, opt.check_lambda);
    unix(command);
    pause(0.1);
    % my_testDropout(name,opt.batchSize,opt.numEpochs, opt.alpha_a, opt.alpha_b, opt.dropout, opt.input_do_rate, opt.hidden_do_rate, opt.Bayesian_do, opt.alpha_lambda_a, opt.alpha_lambda_b, opt.check_lambda)
    %%%%%%%%%%%%%%%
    opt.Bayesian_do = 'LOR';%
    command = sprintf('echo ''matlab -nojvm -r "my_testDropout(''\\''''%s''\\'''',%d,%d,%f,%f,%d,%f,%f,''\\''''%s''\\'''',%f,%f,''\\''''%s''\\'''');quit;"''| qsub -lnodes=1:ppn=2', ...
        name,opt.batchSize, opt.numEpochs, opt.alpha_a, opt.alpha_b, opt.dropout, opt.input_do_rate, opt.hidden_do_rate, ...
        opt.Bayesian_do, opt.alpha_lambda_a, opt.alpha_lambda_b, opt.check_lambda);
    unix(command);
    pause(0.1);
% end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
name = 'nn_gisette';
opt.batchSize = 10;
opt.numEpochs = 2e3;%1e2;%


%%%%%%%%%%%%%%% Without Droput
opt.alpha_lambda_b = 1; 
opt.alpha_lambda_a = 1;
opt.dropout = false;  
filename = filename_writer_nn(name, opt);
filepath = ['/home/ichi/work/Boltzman/Bayesin dropout/DeepLearningDropout/nn/files/'];
load([filepath,filename]);
testErrors=nn.testErrors;
for l=1:length(nn.layers)-1
    lambda_his(l,:) = nn.layers{l}.lambda_history;
end

%%%%%%%%%%%%%%% With Droput
opt.dropout = true; 
opt.Bayesian_do = [];%
filename = filename_writer_nn(name, opt);
filepath = ['/home/ichi/work/Boltzman/Bayesin dropout/DeepLearningDropout/nn/files/'];
load([filepath,filename]);
testErrors_fixed=nn.testErrors;
for l=1:length(nn.layers)-1
    lambda_his_fixed(l,:) = nn.layers{l}.lambda_history;
end

i=1;
for rate = [1e-1 3e-2 1e-2 3e-3 1e-4]
%     for b = [1e2 1e3 1e4]
%         k=1;
        opt.alpha_lambda_b = opt.alpha_b;
        opt.alpha_lambda_a = b*rate;
        
        %%%%%%%%%%%%%%% With Droput
        opt.dropout = true; 
        opt.Bayesian_do = 'UORH';%
        filename = filename_writer_nn(name, opt);
        load([filepath,filename]);
        testErrors_UORH(i,:)=nn.testErrors;
        for l=1:length(nn.layers)-1
            lambda_his_UORH(i,l,:) = nn.layers{l}.lambda_history;
        end
        %%%%%%%%%%%%%%% 
        opt.Bayesian_do = 'LOR';%
        filename = filename_writer_nn(name, opt);
        load([filepath,filename]);
        testErrors_LOR(i,:)=nn.testErrors;
        for l=1:length(nn.layers)-1
            lambda_his_LOR(i,l,:) = nn.layers{l}.lambda_history;
        end
        i=i+1;
% end
end

figure;bar(testErrors_UORH(:,end))
figure(1);
plot(testErrors(50:end));



%%
dodropout=0;
leandropout=0;
a2 =1;b2=1;
i=1;l=1;clear W1 Wt1
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        name = ['woDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
        load(name,'result');
        W1(i,j) = result.valid_accuracy(end);
        Wt1(i,j) = result.test_accuracy(end);
        j = j + 1;
    end
    i = i + 1;
end

dodropout=1;
leandropout=0;
i=1; clear W2 Wt2
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        name = ['wDropout_fixed_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
        load(name,'result');
        W2(i,j) = result.valid_accuracy(end);
        Wt2(i,j) = result.test_accuracy(end);
        j = j + 1;
    end
    i = i + 1;
end

%%
for n=381154:381164 % now running
    command = sprintf('qdel %d', n);
    unix(command);
end