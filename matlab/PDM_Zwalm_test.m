close all
%Van bachelorproef
parameters(1,:) = [400.609,1200,1000]; %cmax
parameters(2,:) = [87.676,100,50]; %cmin
parameters(3,:) = [0.6,0.2,0.5]; %b pareto
parameters(4,:) = [3,2.5,2.5]; % be uitgaande van kwadratisch verband acutele en potentiele
%evapotranspiratie (8) Moore (maar soms toch anders door Cabus blijkbaar)
parameters(5,:) = [8,12,7]; %k1 surface storage
parameters(6,:) = [0.7, 4, 25]; %k2 surface storage
parameters(7,:) = [5.046, 1.7889, 20]; %kb groundwater storage (=> baseflow)
parameters(8,:) = [9000, 21070, 9000]; %kg van soil tension naar groundwater storage
parameters(9,:) = [0.43043,3.6649,0]; %soil tension treshold
parameters(10,:) = [1,1,1]; %bg standaard op 1 (10) Moore
parameters(11,:) = [2,4,2]; %voor 1e 2 timedelay gevonden, niet voor 
%niet voor de 3e :( voorlopig gewoon maar random geval ingevuld (3u)
parameters(12,:) = [0, 0, 0]; %qconst
parameters(13,:) = [0,0,0]; %rainfac niet gebruikt in model
paramPDMhqhcv = parameters(:,1);
A = 109.2300034;

data = readmatrix("../data/Zwalm_data/zwalm_forcings_flow.csv");
P = data(:,2);
P(isnan(P)) = 0;
zwalm.P = P;

EP = data(:,3);
EP(isnan(EP)) = 0;
zwalm.Ep = EP;
zwalm.A = A

Qmod = PDM_Jarne(zwalm, paramPDMhqhcv);

[Qmod_eigen,Qmod_hr, Sb, Qbm3s]  = PDM_eigen_adapted(zwalm, paramPDMhqhcv);
figure()
plot(Qmod)
hold on
plot(Qmod_eigen,'--')

figure()
plot(Qbm3s)


