function [er, bad] = my_testerror(net, x, y, opt)
    if ~isfield(opt,'do_regression')
        do_regression = false;
    else
        do_regression = opt.regression;
    end
    %  feedforward
    if do_regression
      net = feedForward_test_nn_regression(net, x, 1, 1);
      h = net.layers{end}.a;
      a = y;
      bad = [];
      er = mean((h-a).^2);
    else
        if size(net.layers{end}.a,1) == 1
            if isfield(opt,'Bayesian_do')
                if strcmp(opt.Bayesian_do, 'UORH')
                    ido = sig(net.layers{1}.lambda);% opt.input_do_rate(epochNum);
                    hdo = sig(net.layers{2}.lambda);
                    net = feedForward_test_nn(net, x, ido, hdo, opt.dropout);
                else
                    net = feedForward_test_nn(net, x, opt.input_do_rate, opt.hidden_do_rate, opt.dropout);
                end
            else
                net = feedForward_test_nn(net, x, opt.input_do_rate, opt.hidden_do_rate, opt.dropout);
            end
            h = (net.layers{end}.a > 0.5);
            bad = find(h ~= y);
            er = numel(bad) / size(y, 2);
        else
            if isfield(opt,'Bayesian_do')
                if strcmp(opt.Bayesian_do, 'UORH')
                    ido = sig(net.layers{1}.lambda);% opt.input_do_rate(epochNum);
                    hdo = sig(net.layers{2}.lambda);
                    net = feedForward_test_nn(net, x, ido, hdo, opt.dropout);
                else
                    net = feedForward_test_nn(net, x, opt.input_do_rate, opt.hidden_do_rate, opt.dropout);
                end
            else
                net = feedForward_test_nn(net, x, opt.input_do_rate, opt.hidden_do_rate, opt.dropout);
            end
            [~, h] = max(net.layers{end}.a, [], 1);
            [~, a] = max(y, [], 1);
            bad = find(h ~= a);
            er = numel(bad) / size(y, 2);
        end
    end
end