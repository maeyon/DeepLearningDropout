function filename = filename_writer_nn(name, opt)

filename = [name, '_batch',sprintf('%d',opt.batchSize), '_epoch',sprintf('%d',opt.numEpochs)];% sprintf('%1.1e',a)
if opt.dropout
    if strcmp(opt.Bayesian_do, 'UOR')
        filename = [filename,'_UOR_dropout'];
        filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b), '_lambda_a',sprintf('%0.0f',opt.alpha_lambda_a),'_b',sprintf('%0.0f',opt.alpha_lambda_b)];
    elseif strcmp(opt.Bayesian_do, 'UORH')
        filename = [filename,'_UORH_dropout'];
        filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b), '_lambda_a',sprintf('%0.0f',opt.alpha_lambda_a),'_b',sprintf('%0.0f',opt.alpha_lambda_b)];
    elseif strcmp(opt.Bayesian_do, 'LOR')
        filename = [filename,'_LOR_dropout'];
        filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b), '_lambda_a',sprintf('%0.0f',opt.alpha_lambda_a),'_b',sprintf('%0.0f',opt.alpha_lambda_b)];
    elseif strcmp(opt.Bayesian_do, 'FOR')
        filename = [filename,'_FOR_dropout'];
        filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b), '_lambda_a',sprintf('%0.0f',opt.alpha_lambda_a),'_b',sprintf('%0.0f',opt.alpha_lambda_b)];
    else
        filename = [filename,'_fixed_dropout'];
        filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b)];
    end
else
    filename = [filename,'_no_dropout'];
    filename = [filename,'_a',sprintf('%0.0f',opt.alpha_a),'_b',sprintf('%0.0f',opt.alpha_b)];
end

