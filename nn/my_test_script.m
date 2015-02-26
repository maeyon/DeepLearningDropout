addpath ../util;

opt = initializeOptions();
opt.adaptive_alpha = true;%false;%true;
opt.alpha_a = 3e4;%50;%10;%1;
opt.alpha_b = 1e4;

opt.batchSize = 10;
opt.numTestEpochs = 1e4;%1e2;%
 
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

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dodropout=0;
leandropout=0;
a2 =1;b2=1;
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        name = ['woDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
        command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
            name,a,b,a2,b2,dodropout,leandropout);
        unix(command);
        pause(1);
        j = j + 1;
    end
    i = i + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dodropout=0;
leandropout=0;
a2 =1;b2=1;
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        name = ['woDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
        command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
            name,a,b,a2,b2,dodropout,leandropout);
        unix(command);
        pause(1);
        j = j + 1;
    end
    i = i + 1;
end

dodropout=1;
leandropout=0;
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        name = ['wDropout_fixed_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
        command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
            name,a,b,a2,b2,dodropout,leandropout);
        unix(command);
        pause(1);
        j = j + 1;
    end
    i = i + 1;
end
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
i=1;j=2;k=1;l=1;
name = ['woDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
load(name,'result');
acc_MLE = result.test_accuracy;
% time_MLE = result.time;
figure(1);plot(result.test_accuracy(100:end));

i=2;j=2;k=1;l=1;
name = ['wDropout_fixed_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
load(name,'result');
acc_dropout =  result.test_accuracy;
% time_dropout = result.time;
figure(2);plot(result.test_accuracy(100:end));

iter  = 3e7;
kensa = [1:10 20:20:100 200:200:1000 2000:2000:1e4 2e4:2e4:1e5 2e5:2e5:floor(iter)];
figure(3);hold off;
semilogx(kensa, acc_MLE);hold on;
semilogx(kensa, acc_dropout);hold on;
%%
figure(1);
hold off;plot(mean(result.trained_q_all(1:100,:),1));
hold on;plot(mean(result.trained_q_all(101:end,:),1),'r--');

figure(2);
for i=1:6
    subplot(4,3,i);plot(result.trained_q_all(i,:));title([int2str(i)]);ylim([0.4 0.9]);
end
for i=1:6
    subplot(4,3,i+6);plot(result.trained_q_all(i+100,:));title([int2str(i+100)]);ylim([0.4 0.9]);
end

figure(3);plot(result.test_accuracy(100:end));
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
for rate = [0.01 0.02]%[0.01 0.02 0.05]% a = 50 200%[0.2 0.5 1 2 10]
    j=1;
    for b = [2e2 5e2 1e3 2e3]%[200 500 1000]
        k=1;
        a = b*rate;
        for rate2 = [1e-3 2e-3 5e-3 1e-2] 
            l = 1;
            for d = [30 100 300 1e3 3e3]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                load(name,'result');
                W3(i,j,k,l) = result.valid_accuracy(end);
                Wt3(i,j,k,l) = result.test_accuracy(end);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
for rate = [1e-3 2e-3 5e-3 1e-2 2e-2]%[0.01 0.02 0.05]% a = 50 200%[0.2 0.5 1 2 10]
    j=1;
    for b = [1e1 1e2 1e3]%[200 500 1000]
        k=1;
        a = b*rate;
        for rate2 = [1e-3 2e-3 5e-3 1e-2] 
            l = 1;
            for d = [300 1e3 3e3 1e4 3e4]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
                    name,a,b,a2,b2,dodropout,leandropout);
                unix(command);
                pause(1);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
for rate = [1e-3 2e-3 5e-3]%[1e-3 2e-3 5e-3 1e-2 2e-2]%[0.01 0.02 0.05]% a = 50 200%[0.2 0.5 1 2 10]
    j=1;
    for b = [1e1 1e2 1e3]%[200 500 1000]
        k=1;
        a = b*rate;
        for rate2 = [1e-3 2e-3 5e-3 1e-2] 
            l = 1;
            for d = [300 1e3 3e3 1e4 3e4]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                load(name,'result');
                W3(i,j,k,l) = result.valid_accuracy(end);
                Wt3(i,j,k,l) = result.test_accuracy(end);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  All_update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dodropout=1;
leandropout = 1;
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        for rate2 = [3e-4 1e-3 3e-3 1e-2] 
            l = 1;
            for d = [1e3 1e4 1e5]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
                    name,a,b,a2,b2,dodropout,leandropout);
                unix(command);
                pause(1);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Run all experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 100;
n = 100;
alln = 300;
minibatch = 100;
for d = 1:20
    i=1;
    for rate = [1e-1 3e-1 1 3 1e1]
        j=1;
        for b = 1 %[1e2 1e4]
%             k=1;
            a = b*rate;
    %         for rate2 = [3e-4 1e-3 3e-3 1e-2] 
    %             l = 1;
    %             for d = [1e3 1e4 1e5]
    %                 b2 = d*b;
    %                 a2 = rate2*b2;
            a2 = a;
            b2 = b;
            
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %    without dropout 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             dodropout = 0;
%             leandropout = 0;
%             name = ['woDropout_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
%             command = sprintf('echo ''matlab -nojvm -r "learn_test_save2(''\\''''%s''\\'''',%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
%                 name,a,b,minibatch,dodropout,leandropout);
%             unix(command);
%             pause(1);
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %    without dropout 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             dodropout = 1;
%             leandropout = 0;
%             name = ['wDropout_fixed_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
%             command = sprintf('echo ''matlab -nojvm -r "learn_test_save2(''\\''''%s''\\'''',%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
%                 name,a,b,minibatch,dodropout,leandropout);
%             unix(command);
%             pause(1);
%             
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %    with dropout all Learning
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             dodropout = 1;
%             leandropout = 1;
%             name = ['wDropout_allLearning_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
%             command = sprintf('echo ''matlab -nojvm -r "learn_test_save2(''\\''''%s''\\'''',%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
%                 name,a,b,minibatch,dodropout,leandropout);
%             unix(command);
%             pause(1);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %    with dropout single Learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            dodropout = 1;
            leandropout = 1;
            name = ['wDropout_singleLearning_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
            command = sprintf('echo ''matlab -nojvm -r "learn_test_save2(''\\''''%s''\\'''',%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
                name,a,b,minibatch,dodropout,leandropout);
            unix(command);
            pause(1);
    %                 l=l+1;
    %             end
    %             k=k+1;
    %         end
            j = j + 1;
        end
        i = i + 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Load all experiment results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 100;
n = 100;
alln = 300;
minibatch = 100;
for d = 1:20 %:10
    i=1;
    for rate = [1e-3 3e-3 1e-2 3e-2 1e-1]
        j=1;
        for b = 1 %[1e2 1e4]
            a = b*rate;
            a2 = a;
            b2 = b;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %    without dropout 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            dodropout = 0;
            leandropout = 0;
            name = ['woDropout_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
            if exist([name,'.mat'])==2
                load(name,'result');
                W1(i,j,d) = result.valid_accuracy(end);
                Wt1(i,j,d) = result.test_accuracy(end);
                Wbest1(i,j,d) = result.best_test_accuracy;
                Wbestp1(i,j,d) = result.best_possible_test_accuracy;
                Wlength1(i,j,d) = result.last_iter;%length(result.test_accuracy);
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %    without dropout 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            dodropout = 1;
            leandropout = 0;
            name = ['wDropout_fixed_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
            if exist([name,'.mat'])==2
                load(name,'result');
                W2(i,j,d) = result.valid_accuracy(end);
                Wt2(i,j,d) = result.test_accuracy(end);
                Wbest2(i,j,d) = result.best_test_accuracy;
                Wbestp2(i,j,d) = result.best_possible_test_accuracy;
                Wlength2(i,j,d) = result.last_iter;%length(result.test_accuracy);
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %    with dropout all Learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            dodropout = 1;
            leandropout = 1;
            name = ['wDropout_allLearning_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
            if exist([name,'.mat'])==2
                load(name,'result');
                W4(i,j,d) = result.valid_accuracy(end);
                Wt4(i,j,d) = result.test_accuracy(end);
                Wbest4(i,j,d) = result.best_test_accuracy;
                Wbestp4(i,j,d) = result.best_possible_test_accuracy;
                Wlength4(i,j,d) = result.last_iter;%length(result.test_accuracy);
            end

            j = j + 1;
        end
        i = i + 1;
    end
end
%%

[c, index] = max(squeeze(Wt4),[],1);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dodropout=1;
leandropout =1;
N = 100;
n = 100;
alln = 300;
minibatch = 100;
i=4;j=1;d=18;
name = ['woDropout_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
% name = ['wDropout_allLearning_N',int2str(N),'_n',int2str(n),'_alln',int2str(alln),'_d',int2str(d),'=',int2str(i),'_',int2str(j)];
load(name,'result');
% result = learn_test_save2(name,a,b,minibatch,dodropout,leandropout);
%%
dodropout=1;
leandropout = 1;
i=1;
for rate = [3e-4]
    j=1;
    for b = [1e3]
        k=1;
        a = b*rate;
        for rate2 = [3e-3] 
            l = 1;
            for d = [1e3]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_1_2_3_1_long'];
                command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
                    name,a,b,a2,b2,dodropout,leandropout);
                unix(command);
                pause(1);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%
i=1;

name = ['wDropout_allLearning_N2000_1_2_3_1_long'];
load(name,'result');
W3_long = result.valid_accuracy(end);
Wt3_long = result.test_accuracy(end);

[c,index]=max(W3(:));Wt3(index)
[i,j,k,l]=ind2sub(size(W3),index)
%%
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        for rate2 = [3e-4 1e-3 3e-3 1e-2] 
            l = 1;
            for d = [1e3 1e4 1e5]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                load(name,'result');
                W3(i,j,k,l) = result.valid_accuracy(end);
                Wt3(i,j,k,l) = result.test_accuracy(end);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
[c,index]=max(W3(:));Wt3(index)
[i,j,k,l]=ind2sub(size(W3),index)
%%
i=1;j=2;k=3;l=1;
name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
load(name,'result');
acc_all_dropout =  result.test_accuracy;
q_all_dropout =  result.trained_q_all;
time_all_dropout = result.time;
figure(10);plot(result.test_accuracy(100:end));
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Single_update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
dodropout=1;
leandropout = 1;
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        for rate2 = [3e-4 1e-3 3e-3 1e-2] 
            l = 1;
            for d = [1e3 1e4 1e5]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_singleLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                command = sprintf('echo ''matlab -nojvm -r "learn_test_save(''\\''''%s''\\'''',%f,%f,%f,%f,%d,%d);quit;"''| qsub -lnodes=1:ppn=2', ...
                    name,a,b,a2,b2,dodropout,leandropout);
                unix(command);
                pause(1);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%
clear W4 Wt4
i=1;
for rate = [3e-4 1e-3 3e-3 1e-2]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        for rate2 = [3e-4 1e-3 3e-3 1e-2] 
            l = 1;
            for d = [1e3 1e4 1e5]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_singleLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                load(name,'result');
                W4(i,j,k,l) = result.valid_accuracy(end);
                Wt4(i,j,k,l) = result.test_accuracy(end);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
[c,index]=max(W4(:));Wt4(index)
[i,j,k,l]=ind2sub(size(W4),index)
%%
i=1;j=2;k=4;l=1;
name = ['wDropout_singleLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
load(name,'result');
acc_single_dropout =  result.test_accuracy;
q_single_dropout =  result.trained_q;
time_single_dropout = result.time;
figure(4);plot(result.test_accuracy(100:end));
%%
iter  = 3e7;
kensa = [1:10 20:20:100 200:200:1000 2000:2000:1e4 2e4:2e4:1e5 2e5:2e5:floor(iter)];
figure(13);hold off;
semilogx(kensa(1:2:end), acc_MLE(1:2:180),'--c');hold on;
semilogx(kensa(1:2:end), acc_dropout(1:2:180),'x','MarkerSize',12);hold on;
semilogx(kensa(1:2:end), acc_single_dropout(1:2:180),'+k','MarkerSize',12);hold on;
semilogx(kensa(1:2:end), acc_all_dropout(1:2:180),'-r');hold on;
semilogx([1,1e8],0.8413*[1,1],'k-.');
ylim([0.65 0.85]);xlim([1,3e7])
legend('MLE','fixed dropout','single adaptive dropout','all adaptive dropout','Bayes optimal')
%%
figure;bar([acc_MLE(180), acc_dropout(180), acc_single_dropout(180), acc_all_dropout(180)]);
ylim([0.5 0.85])
% acc_single_dropout
%%
figure;plot(1-q_all_dropout(:,end));hold on;
plot([50:100:1000], 1-q_single_dropout(end)*ones(1,10),'+k','MarkerSize',18);
plot([50:100:1000], 1-q_single_dropout(end)*ones(1,10),'+k','MarkerSize',18);
plot([100 100],[-0.1 0.8],'k-.')
xlim([1 1000]);ylim([-0.05 0.8]);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
for rate = [5e-4 1e-3 2e-3]
    j=1;
    for b = [1e2 1e3 1e4]
        k=1;
        a = b*rate;
        for rate2 = [1e-3 2e-3 5e-3 1e-2] 
            l = 1;
            for d = [1e3 3e3 1e4 3e4 1e5]
                b2 = d*b;
                a2 = rate2*b2;
                name = ['wDropout_allLearning_N2000_2',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
                load(name,'result');
                W3(i,j,k,l) = result.valid_accuracy(end);
                Wt3(i,j,k,l) = result.test_accuracy(end);
                l=l+1;
            end
            k=k+1;
        end
        j = j + 1;
    end
    i = i + 1;
end
%%
% i=1;j=3;k=2;l=5;
i=2;j=2;k=3;l=2;
name = ['wDropout_allLearning_N2000_2',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
% name = ['wDropout_allLearning_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
load(name,'result');
%%
figure(21);
hold off;plot(mean(result.trained_q_all(1:100,:),1));
hold on;plot(mean(result.trained_q_all(101:end,:),1),'r--');

figure(22);
for i=1:6
    subplot(4,3,i);plot(result.trained_q_all(i,:));title([int2str(i)]);ylim([0.4 0.9]);
end
for i=1:6
    subplot(4,3,i+6);plot(result.trained_q_all(i+100,:));title([int2str(i+100)]);ylim([0.4 0.9]);
end

figure(23);plot(result.test_accuracy(100:end));


%%

% % clear W1 Wt1
% % i=1;
% % for a = [0.2 0.5 1 2 10]
% %     j=1;
% %     for b = [200 500 1000]
% %         name = ['woDropout_N2000_',int2str(i),'_',int2str(j)];
% %         load(name, 'result');
% %         W1(i,j) = result.valid_accuracy;
% %         Wt1(i,j) = result.test_accuracy;
% %         j = j + 1;
% %     end
% %     i = i + 1;
% % end
% % [c,index]=max(W1(:));Wt1(index)
% 
% dodropout=1;
% leandropout=0;
% a2=1;b2=1;
% i=1;
% for a = [0.2 0.5 1 2 10]
%     j=1;
%     for b = [200 500 1000]
%         name = ['wDropout_N2000_',int2str(i),'_',int2str(j)];
%         command = sprintf('qsh matlab -nodesktop -nojvm -r ''"learn_test_save(''%s'',%f,%f,%f,%f,%d,%d); quit;"''', ...
%             name,a,b,a2,b2,dodropout,leandropout);
%         unix(command);
%         pause(3);
%         j = j + 1;
%     end
%     i = i + 1;
% end
% 
% 
% % clear W2 Wt2
% % i=1;
% % for a = [0.2 0.5 1 2 10]
% %     j=1;
% %     for b = [200 500 1000]
% %         name = ['wDropout_N2000_',int2str(i),'_',int2str(j)];
% %         load(name, 'result');
% %         W2(i,j) = result.valid_accuracy;
% %         Wt2(i,j) = result.test_accuracy;
% %         j = j + 1;
% %     end
% %     i = i + 1;
% % end
% % 
% % [c,index]=max(W2(:));Wt2(index)
% 
% dodropout=1;
% leandropout=1;
% i=1;
% for a = [0.2 0.5 1 2 10]
%     j=1;
%     for b = [200 500 1000]
%         k=1;
%         for c = [0.1 1/3 1 3 10]
%             l = 1;
%             for d = [1e1 1e2 1e3 1e4]
%                 a2 = c*a;
%                 b2 = d*b;
%                 name = ['wDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
%                 command = sprintf('qsh matlab -nodesktop -nojvm -r ''"learn_test_save(''\\''%s\\'''',%f,%f,%f,%f,%d,%d); quit;"''', ...
%                     name,a,b,a2,b2,dodropout,leandropout);
%                 unix(command);
%                 pause(3);
%                 l=l+1;
%             end
%             k=k+1;
%         end
%         j = j + 1;
%     end
%     i = i + 1;
% end
% 
% % clear W3 Wt3
% % i=1;
% % for a = [0.2 0.5 1 2 10]
% %     j=1;
% %     for b = [200 500 1000]
% %         k=1;
% %         for c = [0.1 1/3 1 3 10]
% %             l = 1;
% %             for d = [1e1 1e2 1e3 1e4]
% %                 name = ['wDropout_N2000_',int2str(i),'_',int2str(j),'_',int2str(k),'_',int2str(l)];
% %                 W3(i,j,k,l) = result.valid_accuracy;
% %                 Wt3(i,j,k,l) = result.valid_accuracy;
% %                 l=l+1;
% %             end
% %             k=k+1;
% %         end
% %         j = j + 1;
% %     end
% %     i = i + 1;
% % end
% % 
% % [c,index]=max(W3(:));Wt3(index)
% 
% %
%%
for n=368534:368577 % now running
    command = sprintf('qdel %d', n);
    unix(command);
end