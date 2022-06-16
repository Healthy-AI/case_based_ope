function [] = create_dataset(workspace_path, results_path)
    % Load data
    load(workspace_path)

    % Run script
    addpath('./src/case_based_ope/sepsis/ai_clinician/');
    addpath('./src/case_based_ope/sepsis/ai_clinician/MDPtoolbox');
    AIClinician_core_160219;

    % Identify best policy
    softbestpol = abs(zeros(752,25)-p/24);
    for i = 1:750
        optact = OA(:, bestpol);
        softbestpol(i,optact(i)) = 1-p;
    end

    % Soft physician's policy
    softpi = physpol;
    for i = 1:750
        ii = softpi(i, :) == 0;
        z = p/sum(ii);
        nz = p/sum(~ii);
        softpi(i, ii) = z;
        softpi(i, ~ii) = softpi(i, ~ii) - nz;
    end

    % Extract states
    states = knnsearch(C, MIMICzs);

    % Save data
    aic = strcat(results_path, 'ai_clinician/');
    if exist(aic, 'dir') == 0
        mkdir(aic);
    end

    writematrix(softbestpol, strcat(aic, 'target_policy.csv'))
    writematrix(softpi, strcat(aic, 'behavior_policy.csv'))
    writematrix(train, strcat(aic, 'is_train.csv'))
    writematrix(qldata3train, strcat(aic, 'qldata_train.csv'))
    writematrix(qldata3test, strcat(aic, 'qldata_test.csv'))
    writematrix(states, strcat(aic, 'states.csv'))
end