function [er, bad, std_er] = my_testerror_dropout(net, x, y, numTestEpochs, do_regression)
    %  feedforward
    if ~exist('do_regression','var')
        do_regression=false;
    end

    A = arrayfun(@(i) my_feedForward_test_dropout_nn(net, x), 1:numTestEpochs, 'UniformOutput', false);
    s = size(A{1,1});
    A = cell2mat(A);
    A = reshape(A, [s, numTestEpochs]);
    
    a = mean(A, 3);
    if nargout > 2
        std_er = std(A,0,3);
    end

    
    if do_regression
      est = a;
      real = y;
      bad = [];
      er = mean((est-real).^2);
    else
        if size(a,1) == 1
          h = (a > 0.5);
          bad = find(h ~= y);
          er = numel(bad) / size(y, 2);
        else
          [~, est] = max(a,[],1);
          [~, real] = max(y,[],1);
          bad = find(est ~= real);
          er = numel(bad) / size(y, 2);
        end
    end
end