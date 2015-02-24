function net = setup_nn(net, x , y, opt)
	numNodePrev = size(x,1);
    net.iter = 0;
    net.nodeRanges = zeros(length(net.layers), 2);
    net.nodeRanges(1,:) = [1, numNodePrev];
    if isfield(opt, 'Bayesian_do')
        if strcmp(opt.Bayesian_do, 'UOR') % 'UOR'(uniformly optimized rate dropout)
            net.layers{1}.lambda = invsig(opt.input_do_rate); 
        elseif strcmp(opt.Bayesian_do, 'LOR') % 'LOR'(layer-wise optimized rate dropout)
            net.layers{1}.lambda = invsig(opt.input_do_rate);
        elseif strcmp(opt.Bayesian_do, 'FOR') % 'FOR'(feature-wise optimized rate dropout)
            net.layers{1}.lambda = repmat(invsig(opt.input_do_rate), numNodePrev, 1);
        else
            net.layers{1}.lambda = invsig(opt.input_do_rate);
        end
        
        net.delta = 1/size(x,2);
    else
        net.layers{1}.lambda = invsig(opt.input_do_rate);
    end
	for l = 2:length(net.layers)-1
		net.layers{l}.w = rand(net.layers{l}.n, numNodePrev) * 2 - 1; % Initialize weight matrix - value between -1 and 1
		net.layers{l}.b = zeros(net.layers{l}.n,1); % Initialize bias - need to experiment with different methods
		numNodePrev = net.layers{l}.n;
        net.nodeRanges(l,:) = [net.nodeRanges(l-1,2)+1, net.nodeRanges(l-1,2)+numNodePrev];
        if isfield(opt, 'Bayesian_do')
            if strcmp(opt.Bayesian_do, 'UOR') % 'UOR'(uniformly optimized rate dropout)
                net.layers{l}.lambda = invsig(opt.hidden_do_rate);
            elseif strcmp(opt.Bayesian_do, 'LOR') % 'LOR'(layer-wise optimized rate dropout)

            elseif strcmp(opt.Bayesian_do, 'FOR') % 'FOR'(feature-wise optimized rate dropout)
                net.layers{l}.lambda = opt.hidden_do_rate* numNodePrev ;
            else
                net.layers{l}.lambda = invsig(opt.hidden_do_rate);
            end
        else
            net.layers{l}.lambda = invsig(opt.hidden_do_rate);
        end
	end
	l = length(net.layers);
	net.layers{l}.w = rand(size(y,1),numNodePrev) * 2 - 1; % Initialize weight matrix - need to experiment with different methods
	net.layers{l}.b = zeros(size(y,1),1);
	net.layers{1}.do = ones(1) * -1;
    net.nodeRanges(l,:) = [net.nodeRanges(l-1,2)+1, net.nodeRanges(l-1,2)+size(y,1)];
end