VthMin = optimizableVariable('VthMin', [1e-3, 2e-2], 'Type', 'real');
VthMax = optimizableVariable('VthMax', [5e-2, 1.0], 'Type', 'real');
Ve = optimizableVariable('Ve', [1e-3, 5e-3], 'Type', 'real');
tau = optimizableVariable('tau', [1e-1, 10], 'Type', 'real');
R = optimizableVariable('R', [1e2, 5e3], 'Type', 'real');
Target = optimizableVariable('Target', [2e1, 4e1], 'Type', 'real');
gamma = optimizableVariable('gamma', [5e-8, 5e-5], 'Type', 'real');
refPer = optimizableVariable('refPer', [0, 2e-1], 'Type', 'real');
inhib = optimizableVariable('inhib', [1e-9, 1e-3], 'Transform', 'log');
objfun=@(x) Main_Routine(x, false);
init_params = array2table([0.013571, 0.33874, 0.0010161, 6.0145, 3614.5, 24.924, 3.2019e-06, 0.17555, 1.1687e-07],...
                          'VariableNames', {'VthMin', 'VthMax', 'Ve', 'tau', 'R', 'Target', 'gamma', 'refPer', 'inhib'});
result = bayesopt(objfun, [VthMin, VthMax, Ve, tau, R, Target, gamma, refPer, inhib], 'IsObjectiveDeterministic',true,...
    'PlotFcn', {@plotMinObjective,@plotConstraintModels},...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',0, 'MaxObjectiveEvaluations', 250,...
    'InitialX', init_params, 'UseParallel',true);
best_params = result.XAtMinObjective;
disp(best_params);
Main_Routine(best_params, true);


%      VthMin     VthMax        Ve         tau        R       Target      gamma       refPer       inhib   
%     ________    _______    _________    ______    ______    ______    __________    _______    __________
% 
%     0.013571    0.33874    0.0010161    6.0145    3614.5    24.924    3.2019e-06    0.17555    1.1687e-07