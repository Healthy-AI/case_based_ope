function [] = create_dataset(data_path)
    % Import data
    abx = table2array(readtable(strcat(data_path, 'raw/', 'abx.csv')));
    culture = table2array(readtable(strcat(data_path, 'raw/', 'culture.csv')));
    microbio = table2array(readtable(strcat(data_path, 'raw/', 'microbio.csv')));
    demog = (readtable(strcat(data_path, 'raw/', 'demog.csv')));
    ce010 = table2array(readtable(strcat(data_path, 'raw/', 'ce010000.csv')));
    ce1020 = table2array(readtable(strcat(data_path, 'raw/', 'ce1000020000.csv')));
    ce2030 = table2array(readtable(strcat(data_path, 'raw/', 'ce2000030000.csv')));
    ce3040 = table2array(readtable(strcat(data_path, 'raw/', 'ce3000040000.csv')));
    ce4050 = table2array(readtable(strcat(data_path, 'raw/', 'ce4000050000.csv')));
    ce5060 = table2array(readtable(strcat(data_path, 'raw/', 'ce5000060000.csv')));
    ce6070 = table2array(readtable(strcat(data_path, 'raw/', 'ce6000070000.csv')));
    ce7080 = table2array(readtable(strcat(data_path, 'raw/', 'ce7000080000.csv')));
    ce8090 = table2array(readtable(strcat(data_path, 'raw/', 'ce8000090000.csv')));
    ce90100 = table2array(readtable(strcat(data_path, 'raw/', 'ce90000100000.csv')));
    labU = [
        table2array(readtable(strcat(data_path, 'raw/', 'labs_ce.csv')));
        table2array(readtable(strcat(data_path, 'raw/', 'labs_le.csv')))
    ];
    MV = table2array(readtable(strcat(data_path, 'raw/', 'mechvent.csv')));
    inputpreadm = table2array(readtable(strcat(data_path, 'raw/', 'preadm_fluid.csv')));
    inputMV = table2array(readtable(strcat(data_path, 'raw/', 'fluid_mv.csv')));
    inputCV = table2array(readtable(strcat(data_path, 'raw/', 'fluid_cv.csv')));
    vasoMV = table2array(readtable(strcat(data_path, 'raw/', 'vaso_mv.csv')));
    vasoCV = table2array(readtable(strcat(data_path, 'raw/', 'vaso_cv.csv')));
    UOpreadm = table2array(readtable(strcat(data_path, 'raw/', 'preadm_uo.csv')));
    UO = table2array(readtable(strcat(data_path, 'raw/', 'uo.csv')));
    
    % Run scripts
    addpath('./src/case_based_ope/sepsis/ai_clinician/');
    load('reference_matrices.mat');
    AIClinician_sepsis3_def_160219;
    AIClinician_mimic3_dataset_160219;

    % Save data
    processed = strcat(data_path, 'processed/');
    if exist(processed, 'dir') == 0
        mkdir(processed);
    end
    writetable(MIMICtable, strcat(processed, 'sepsis_data.csv'), 'Delimiter', ',');
    
    interim = strcat(data_path, 'interim/');
    if exist(interim, 'dir') == 0
        mkdir(interim);
    end
    save(strcat(interim, 'ai_clinician_workspace.mat'));
end