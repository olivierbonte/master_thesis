%% Algemene test van de PDM
load('data/inputs.mat')
load('data/paramPDM.mat')
uren = length(inputs.P);
[Q_out, Q_out_hr, Sb] = PDM_eigen(inputs,paramPDM); %input op uurbasis
[Qout_adapted, Q_out_hr_adapted, Sb_adapted] = PDM_eigen_adapted(inputs, paramPDM);
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
plot(Sb)
%figure()
%plot(inputs.P)
writematrix(Q_out,'output/Qmod.csv')
writematrix(Q_out_hr, 'output/Qmod_hr.csv')
writematrix(Q_out_hr_adapted, 'output/Qmod_adapted_hr.csv')

