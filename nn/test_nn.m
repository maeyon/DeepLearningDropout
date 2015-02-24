function nn = test_nn(opt,nn);

    %%
    load(opt.dataset);
    train_x = reshape(train_x, size(train_x,1), size(train_x,3));
    test_x = reshape(test_x, size(test_x,1), size(test_x,3));
    %% 

    if ~isfield(opt,'randstate')
        opt.randstate=0;
    end
    rand('state',opt.randstate)

    if nargin < 2
        nn.layers = opt.layers;

%         nn.testErrors = zeros(opt.numEpochs,1);
%         nn.testErrorsDropout = zeros(opt.numEpochs,1);
%         nn.trainingErrors = zeros(opt.numEpochs,1);
        nn = setup_nn(nn, train_x, train_y, opt);
    end
    
%     nn = setup_nn(nn, train_x, train_y(1,:));
%     nn = train_nn(nn, train_x, train_y(1,:), test_x, test_y(1,:), opt);
    
    nn = train_nn(nn, train_x, train_y, test_x, test_y, opt);

end

