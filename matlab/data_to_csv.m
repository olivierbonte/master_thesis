load('data/inputs.mat')
load('data/paramPDM.mat')
%celldata = struct2cell(inputs);
%P_EP_table = cell2table(celldata([1,2]));
%writetable(P_EP_table,'data/inputs_P_EP.csv')
%writema
writematrix(inputs.P,'data/P.csv')
writematrix(inputs.Ep,'data/EP.csv')
writematrix(inputs.observations, 'data/observations.csv')
writematrix(inputs.A, 'data/Area.csv')
writematrix(paramPDM, 'data/paramPDM.csv')

