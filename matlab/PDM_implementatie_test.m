%% Algemene test van de PDM
load('data/inputs.mat')
load('data/paramPDM.mat')
uren = length(inputs.P);
Q_out = PDM(inputs,paramPDM); %input op uurbasis, output op dagbasis
dagen = length(Q_out);
Q_obs = inputs.observations;

close all
figure()
plot(1:dagen,Q_out, 'b', 1:dagen,Q_obs, 'r')
xlabel('dagen')
ylabel('Debiet: m^3/s')
legend('PDM voorspelling','observatie')
%bemerkt de soms grote uitschieters van PDM tov de geobserveerde
%waarnemingen

figure()
plot(inputs.P)
writematrix(Q_out,'output/Qmod.csv')

