function net = my_backPropagation_nn(net, y, opt)
epochNum = net.iter;
numLayers = length(net.layers);
e = net.layers{numLayers}.a - y; % Total error

if opt.regression
    net.layers{numLayers}.d = e .* (net.layers{numLayers}.a .* (1 - net.layers{numLayers}.a));
    negL = 1/2*e(:).^ 2; % sample-wise negative log-likelihood
    net.L = mean(negL);% 1/2* sum(e(:) .^ 2) / size(e, 2); % Mean-squared loss for future checking
else
    net.layers{numLayers}.d = e;
    logy = log(net.layers{numLayers}.a);
    log1_y = log(1-net.layers{numLayers}.a);
    negL = 0; % sample-wise negative log-likelihood
    for i = 1:size(y,1)
        negL = negL - y(i,:).*logy(i,:) - (1-y(i,:)).*log1_y(i,:); % Cross entropy
    end
    net.L = mean(negL);%net.L/size(e, 2);
end

% Compute delta values for each layer
for l = (numLayers - 1) : -1 : 1
    if opt.gaussian
        grad = net.layers{l}.ga .* (net.layers{l}.a .* (1 - net.layers{l}.a));
    else
        grad = (net.layers{l}.a .* (1 - net.layers{l}.a));
    end
    if opt.dropconnect
        net.layers{l}.d = (net.layers{l + 1}.wdc' * net.layers{l + 1}.d) .* grad;
    else
        net.layers{l}.d = (net.layers{l + 1}.w' * net.layers{l + 1}.d) .* grad;
    end
end

% Perform gradient descent, no weights for max-pooling layer
if opt.adaptive_alpha
    alpha = opt.alpha_a/(opt.alpha_b + epochNum);
    for l = 2 : numLayers
        net.layers{l}.b = net.layers{l}.b - alpha * sum(net.layers{l}.d,2) / size(net.layers{l}.d,2);
        net.layers{l}.w = net.layers{l}.w - alpha * net.layers{l}.d * net.layers{l - 1}.a' / size(net.layers{l}.d,2);
        if opt.dropconnect
            net.layers{l}.wdc = net.layers{l}.w .* net.layers{l-1}.dc;
        end
    end
else
    for l = 2 : numLayers
        net.layers{l}.b = net.layers{l}.b - opt.alpha * sum(net.layers{l}.d,2) / size(net.layers{l}.d,2);
        net.layers{l}.w = net.layers{l}.w - opt.alpha * net.layers{l}.d * net.layers{l - 1}.a' / size(net.layers{l}.d,2);
        if opt.dropconnect
            net.layers{l}.wdc = net.layers{l}.w .* net.layers{l-1}.dc;
        end
    end
end

%%%%%%%%%%%%%%%%% Update dropout rate
if isfield(opt, 'Bayesian_do') && opt.dropout
    ido = sig(net.layers{1}.lambda);% opt.input_do_rate(epochNum);
    for l = 2:length(net.layers)-1
        hdo{l} = sig(net.layers{l}.lambda);%opt.hidden_do_rate(epochNum);
    end
    
    %%%%%%%%%%%%%%% update dropout rate in input layer 
    grad = 0;
    if ~isfield(opt,'adaptive_alpha_lambda')
        opt.adaptive_alpha_lambda = false;
    end
    if ~isfield(opt,'alpha_lambda')
        opt.alpha_lambda = opt.alpha;
    end
    if strcmp(opt.Bayesian_do, 'UOR')
        temp = sum(net.layers{1}.do- ido,1);
        Nunits = size(net.layers{2}.w, 2);
        for l = 2:length(net.layers)-1
            temp = temp + sum(net.layers{l}.do-hdo{l},1) ;
            Nunits = Nunits + net.layers{l}.n;
        end
        logratio = log(opt.input_do_rate) - log(1-opt.input_do_rate);
        grad = mean(temp.*negL) - net.delta*(ido*(1-ido)*(log(1-ido)-log(ido)+logratio))*Nunits;
        if opt.adaptive_alpha_lambda
            alpha_lambda = opt.alpha_lambda_a/(opt.alpha_lambda_b + epochNum);
            net.layers{1}.lambda = net.layers{1}.lambda - alpha_lambda*grad;
        else
            net.layers{1}.lambda = net.layers{1}.lambda - opt.alpha_lambda*grad;
        end
        for l = 2:length(net.layers)-1
            net.layers{l}.lambda = net.layers{1}.lambda;
        end
    elseif strcmp(opt.Bayesian_do, 'UORH')
        logratio = log(opt.input_do_rate) - log(1-opt.input_do_rate);
        grad = mean((sum(net.layers{1}.do- ido,1) ).*negL) ...
            - net.delta*(ido*(1-ido)*(log(1-ido)-log(ido)+logratio))*size(net.layers{2}.w, 2);
        temp = 0;
        Nunits = 0;
        for l = 2:length(net.layers)-1
            temp = temp + sum(net.layers{l}.do-hdo{l},1);
            Nunits = Nunits + net.layers{l}.n;
        end
        logratio = log(opt.hidden_do_rate) - log(1-opt.hidden_do_rate);
        grad2 = mean(temp.*negL) - net.delta*( hdo{2}*(1- hdo{2})...
            *(log(1- hdo{2})-log(hdo{2}) + logratio))*Nunits;
%         if epochNum > 243*400-1 && epochNum < 244*400
%             fprintf('kita\n');
%             dbstop at 97 in my_train_nn.m;
%         end
        if opt.adaptive_alpha_lambda
            alpha_lambda = opt.alpha_lambda_a/(opt.alpha_lambda_b + epochNum);
            net.layers{1}.lambda = net.layers{1}.lambda - alpha_lambda*grad;% 1e-5
            net.layers{2}.lambda = net.layers{2}.lambda - alpha_lambda*grad2;% 1e-5
        else
            net.layers{1}.lambda = net.layers{1}.lambda - opt.alpha_lambda*grad;
            net.layers{2}.lambda = net.layers{2}.lambda - opt.alpha_lambda*grad2;
        end
        for l = 2:length(net.layers)-1
            net.layers{l}.lambda = net.layers{2}.lambda;
        end
    elseif strcmp(opt.Bayesian_do, 'LOR')
        logratio = log(opt.input_do_rate) - log(1-opt.input_do_rate);
        grad(1) = mean((sum(net.layers{1}.do-ido,1) ).*negL) ...
            - net.delta*(ido*(1-ido)*(log(1-ido)-log(ido)+logratio))*size(net.layers{2}.w, 2);
        logratio = log(opt.hidden_do_rate) - log(1-opt.hidden_do_rate);
        for l = 2:length(net.layers)-1
            grad(l) = mean(( sum(net.layers{l}.do -  hdo{l},1)).*negL) ...
                - net.delta*( hdo{l}*(1- hdo{l})*(log(1- hdo{l})-log(hdo{l})+logratio))*net.layers{l}.n;
        end

        if opt.adaptive_alpha_lambda
            alpha_lambda = opt.alpha_lambda_a/(opt.alpha_lambda_b + epochNum);
            for l=1:length(net.layers)-1
                net.layers{l}.lambda = net.layers{l}.lambda - alpha_lambda*grad(l);
            end
        else
            for l=1:length(net.layers)-1
                net.layers{l}.lambda = net.layers{l}.lambda - opt.alpha_lambda*grad(l);
            end
        end
    end
    
    %%%%%%%%%%%%%%% update dropout rate in hidden layers 
end

end