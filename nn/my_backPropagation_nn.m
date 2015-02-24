function net = my_backPropagation_nn(net, y, opt)
epochNum = net.iter;
numLayers = length(net.layers);
e = net.layers{numLayers}.a - y; % Total error

if opt.regression
    net.layers{numLayers}.d = e .* (net.layers{numLayers}.a .* (1 - net.layers{numLayers}.a));
    net.L = 1/2* sum(e(:) .^ 2) / size(e, 2); % Mean-squared loss for future checking
else
    net.layers{numLayers}.d = e;
    logy = log(net.layers{numLayers}.a);
    log1_y = log(1-net.layers{numLayers}.a);
    net.L = 0;
    for i = 1:size(y,1)
        net.L = net.L - y(i,:)*logy(i,:)' - (1-y(i,:))*log1_y(i,:)'; % Cross entropy
    end
    net.L = net.L/size(e, 2);
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

if isfield(opt, 'Bayesian_do')
    ido = sig(net.layers{1}.lambda);% opt.input_do_rate(epochNum);
    for l = 2:length(net.layers)-1
      hdo{l} = sig(net.layers{l}.lambda);%opt.hidden_do_rate(epochNum);
    end
    
    %%%%%%%%%%%%%%% update dropout rate in input layer 
    grad = 0;
    if strcmp(opt.Bayesian_do, 'UOR')
        grad = mean(sum(net.layers{1}.do,1) -  ido)*net.L - net.delta*(ido.*(1-ido).*(log(1-ido)-log(ido)));
    end
    
    if opt.adaptive_alpha
        net.layers{1}.lambda = net.layers{1}.lambda - alpha*grad;
    else
        net.layers{1}.lambda = net.layers{1}.lambda - opt.alpha*grad;
    end
    
    for l = 2:length(net.layers)-1
        if strcmp(opt.Bayesian_do, 'UOR')
            grad = mean(sum(net.layers{l}.do,1) -  ido)*net.L - net.delta*(hdo{l}.*(1-hdo{l}).*(log(1-hdo{l})-log(hdo{l})));
        end

        if opt.adaptive_alpha
            net.layers{l}.lambda = net.layers{l}.lambda - alpha*grad;
        else
            net.layers{l}.lambda = net.layers{l}.lambda - opt.alpha*grad;
        end
    end
    %%%%%%%%%%%%%%% update dropout rate in hidden layers 
end

end