function net = my_feedForward_test_nn(net, x, opt)
% Instead of dropping out nodes, each weight is multiplied by the dropout
% rate. To be used on test data.

	numLayers = length(net.layers); % total number of layers
	net.layers{1}.a = x;
	for l = 2:numLayers
        if opt.dropout
            if strcmp(opt.Bayesian_do, 'UOR') || strcmp(opt.Bayesian_do, 'UORH') || strcmp(opt.Bayesian_do, 'LOR')
                hdo = sig(net.layers{l-1}.lambda);
                net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a * hdo, net.layers{l}.b));
            elseif strcmp(opt.Bayesian_do, 'FOR')
                hdo = sig(net.layers{l-1}.lambda);
                net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a.* hdo, net.layers{l}.b));
            else
                if l == 2
                    net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a * opt.input_do_rate, net.layers{l}.b));
                else
                    net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a * opt.hidden_do_rate, net.layers{l}.b));
                end
            end
        else
            net.layers{l}.a = sigmoid(bsxfun(@plus, net.layers{l}.w * net.layers{l - 1}.a, net.layers{l}.b));
        end
	end
end