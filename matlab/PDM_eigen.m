function [Qmod] = PDM_eigen(inputs,X)


 P=inputs.P; 
 Ep=inputs.Ep; % de potentiële evapotranspiratie (Ei' van Moore)
 A=inputs.A; %Oppervlakte van het stroomgebied (ik denk in km^2),
 %welke oppervlakte moeten we gebruiken hiervoor? Want normaal is dat van
 %één stroomgebied maar nu werken we met de 3 groepen...
 
%P: hourly rainfall [mm/h];
%Ep: hourly evapotranspiration [mm/h];
%X: array of parameters;

%% definitie matrixdimensies
uren=length(P);

Ea      =zeros(uren,1); %mm, de actuele evapotranspiratie (niet potentieel)
Qd      =zeros(uren,1); %mm/u, de direct run off (naar oppervalkte opslag)
D       =zeros(uren,1); %mm/u drainage naar grondwateropslag
C       =zeros(uren,1); %mm store capacity (kleine c Moore)
S1      =zeros(uren,1); %mm opslag obv bodemspanning
pi      =zeros(uren,1); %mm de netto (effectieve dixit james) neerslag
Qb      =zeros(uren,1); %mm/u flow UIT DE grondwateroplsag
Qbm3s   =zeros(uren,1); %m3/s " " " " in m^3
of1     =zeros(uren,1); %mm/u niet gebruikt
of1m3s  =zeros(uren,1); %m3/s " " 
of2     =zeros(uren,1); %mm/u " "
of2m3s  =zeros(uren,1); %m3/s omzeeting qr in m^3/s 
Qmod    =zeros(uren,1); %m3/s de totale flow = DE output van ons model

% extra Pieter
Sb      = zeros(uren,1); %mm
qr      = zeros(uren,1); %mm/u de surface run off UIT oppervlakte opslag 
% volgens de '2 bakjes'

%% Load parameters
cmax=X(1); % minimum soil moisture opslag (mm)
cmin=X(2); % maximumu soil moisture opslag (mm)
b=X(3); % cte voor pareto distributie
be=X(4); % voor de verhouding Ei (actueel evapotrans) op Ei' (potentiele) 
k1=X(5); % voor surface runoff (vanuit opp reservoir) (u)
k2=X(6); % ''
kb=X(7); % voor routing VAN grondwateropslag (dus voor 'baseflow' berekening) (u/mm^2)
kg=X(8); % voor recharge NAAR grondwaterslag (u)
St=X(9); % Soil tension storage capacity (mm water die bodem vasthouden kan!) (mm)
% is dus een treshold
bg=X(10); % voor recharge functie NAAR grondwateropslag (10) Moore
tdly=ceil(X(11)); %Time delay, wordt gebruikt voor Qmod, hoe werkt dit??? is 
% dit verschil tussen regenvallen en effect in flow??? 
%in paramPDM is deze 7 en aangezien 1 tijdstap 1 uur dus een vertraging van
%7u veronderstel ik
qconst=X(12); %toevoegen van een constante flow (naast qr en qb) (m^3/s)
rainfac=X(13); % alpha, bepaalt hoeveel naar oppervlakte opslag en hoeveel naar
% grondwater opslag als NIET naar soil moisutre (bodemspanning), niet
% gebruikt eigenlijk...

Smax=((b*cmin)+cmax)/(b+1); % zie Appendix A pareto (3)
%maximale hoeveelheid water vastgehouden op bodemspanning
deltat=1;               %tijdstap van 1 uur! cruciaal!
Am2=A*1000*1000; % oppervlakte stroomgebied in m^2

% extra initialisatie Pieter (implementatie 2 lineaire stores)
%zie formules (26) en (27) Moore
delta1x = exp(-deltat/k1); 
delta2x = exp(-deltat/k2);
delta1  =-(delta1x + delta2x);
delta2  = delta1x * delta2x;
%formules wanneer k1 en k2 niet gelijk zijn
omega0  = (k1*(delta1x-1)-k2*(delta2x-1))/(k2-k1);
omega1  = ((k2*(delta2x-1)*delta1x)-(k1*(delta1x-1)*delta2x))/(k2-k1);
Sb(1)   = 0.001;

start=1;
i=start;                    %tijdstap 1

%initiële wateropslag (schatting!)
X(14)=Smax/2;
S1(i)=X(14);


%berekening C*(1)(= de kritische soil moisture capacity),
%Ea(1) (acutele evapotranspiratie),
%D(1) (drainage naar grondwater opslag)
%en pi(1) (effectieve regen) voor de eerste tijdstap
C(i)=cmin+(cmax-cmin)*(1-((Smax-S1(i))/(Smax-cmin))^(1/b+1)); %(4) Pareto  appendix A
%cmax*(1-(1-S1(i)/Smax)^(1/(b+1))) (20b) wanneer enkel cmax (geen cmin)
if C(i)>cmax
    C(i)=cmax;
elseif C(i)<=0 %vraag: dus c kan lager dan c_min gaan? Antwoord: p.487 
    % in droge periodes,  blijft vaag...
    C(i)=0;
end;

Ea(i)=Ep(i,1)*(1-((Smax-S1(i))/Smax)^be); % (8) Moore, relatie act en pot 
if S1(i)>St %St = tresholdwaarde = hoeveel water de bodem onder spanning 
    %vasthouden kan (hier onder => geen water naar grondwateropslag)
        D(i)=(1/kg)*((S1(i)-St)^bg); % (10) Moore
else
        D(i)=0;
end;
pi(i)=P(i,1)-Ea(i)-D(i); %(14) Moore: netto regen = regen gevallen - 
% de acutele evapotranspiratie - naar grondwater reservoir 

%berekeningen voor elke volgende tijdstap i
i=start+1;
imax=uren;
for i=i:imax
    %actuele evapotranspiratie
    Ea(i)=Ep(i,1)*(1-((Smax-S1(i-1))/Smax)^be); %(8)
    %drainage naar grondwaterreservoir
    if S1(i-1)>St
        D(i)=(1/kg)*((S1(i-1)-St)^bg);
    else
        D(i)=0;
    end
    %netto neerslagoverschot
    pi(i)=P(i,1)-Ea(i)-D(i);
    %directe runoff Qd
    voorwaarde=0;
    % volstaat de voorwaarde P(i,1)>0 (Bruno) of moet de nettoneerslag ook
    % groter zijn dan 0
    %if (P(i,1)>0)
    if (pi(i)>0) %dus de NETTO neerslag (of overschot) dus groter dan 0!
        voorwaarde=voorwaarde+1;
    end;
    if (C(i-1)>cmin) %enkel dan is er run off!!
        voorwaarde=voorwaarde+1;
    end;
    if voorwaarde==2 %beide voorwaarden dus voldaan
        %Qd(i)=(pi(i))*(1-((1-((C(i-1)-cmin)/(cmax-cmin)))^b));% hoe komt hij aan deze vergelijking ?
        % Pieter en Els deden een poging de integraal op te lossen (vgl.6
        % +vgl 5 (Moore)
        if (C(i-1)+pi(i))< cmax 
            %hieronder (6) met delta t = 1 uit Appendix A
            % V(t+delta t) = pi*1 + (S(t+delta t) - S(t))
            %hierbij worden S(t) en S(t+ delta t) geïmplementeerd volgens
            %(4) Appendix A. Bemerk dat...
            %...steunend op (3) geldt dat \bar{c} - cmin = (cmax-cmin)/(b+1)
            %...C(i-1) = C*(t)
            %... obv (5) Moore: C*(t + deltat) = C(t) + pi 
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1)-((cmax-C(i-1)-pi(i))/(cmax-cmin))^(b+1));
        % als de nettoneerslag zorgt voor een volledige verzadiging ...
        else %Dus groter dan cmax nu => ... Nog wat vragen...
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1))+(C(i-1)+pi(i)-cmax);
        end
    else
        Qd(i)=0; % er is dus geen directe run off naar het opp reservoir
    end;
    if Qd(i)<0
        Qd(i)=0;
    end;
    %opslag in onverzadigde zone
    S1(i)=S1(i-1)+(pi(i))-Qd(i);      %mag niet groter worden dan Smax
    %Idee van (17) Moore (zie uitleg p. 486 Delta(S + deltat) = pi*dt -
    %V(t+deltat))
    if S1(i)>Smax
        S1(i)=Smax;
    elseif S1(i)<0                      %mag niet negatief worden
        S1(i)=0;
    end;
    %C
    C(i)=cmin+((cmax-cmin)*(1-(((Smax-S1(i))/(Smax-cmin))^(1/(b+1)))));
    % formule (5) Appendix A
    if C(i)>cmax
        C(i)=cmax;
    elseif C(i)<=0
        C(i)=0;
    end;
    %basisafvoer (Qb in mm/u)
    %Qb(i)=(D(i)*deltat/(kb+0.5*deltat))+(Qb(i-1)*(kb-0.5*deltat)/(kb+0.5*deltat)); %cfr. RVL
   %?
   
    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%
    % Sb is dus opslag in grondwateropslag
    Sb(i) = Sb(i-1) - (1/(3*kb*(Sb(i-1)^2)))*(exp(-3*kb*(Sb(i-1)^2))-1)*(D(i)-kb*(Sb(i-1)^3));
    %(24) Moore (o.b.v. Horton Izzard)
    Qb(i) = kb*(Sb(i)^3);
    %(25) Moore, Qb is de baseflow (weg UIT grondwateropslag)in mm/u
    if Qb(i)<0
        Qb(i)=0;
    end;
    Qbm3s(i)=Qb(i)/1000*Am2/3600; %/1000 van mm naar m, * oppervlak van het
    %stroomgebied in m^2, 3600 van u naar s
    %oppervlakkige afvoer (cfr RVL) (of2 in mm/u)
    %of1(i)=(Qd(i)*deltat/(k1+0.5*deltat))+(of1(i-1)*(k1-0.5*deltat)/(k1+0.5*deltat)); %cfr. RVL
    %of1m3s(i)=of1(i)/1000*Am2/3600;
    %of2(i)=(of1(i)/(k2+0.5*deltat))+(of2(i-1)*(k2-0.5*deltat)/(k2+0.5*deltat)); %cfr. RVL
    %of2m3s(i)=of2(i)/1000*Am2/3600;
   
    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%
    % Dit is dus de surface runoff UIT de grondwateropslag volgens transfer
    % functie model ('de 2 bakjes')!
    if i > 2
    qr(i)=-delta1*qr(i-1)-delta2*qr(i-2)+omega0*Qd(i)+omega1*Qd(i-1); %(26) Moore
    of2m3s(i)=qr(i)/1000*Am2/3600; %analoog 
    end
   
    %totale afvoer
    Qmod(i+tdly,1)=Qbm3s(i)+of2m3s(i)+qconst;
    %praktische vraag: doordat tdly = 7 is er geen risico van hoger getal
    %voor index te bekomen dan mogelijk in Qmod? 
    
   
   
%     if mod(i,100000)==0
%         Qmm=Qmod.*1000.*3600./Am2;
%         PDMres=[T P D pi Ep Ea Qd Qb Qmm S1./Smax];
%         save('tmp_PDM_output','PDMres');
%     end
   
end;

 Qmod=Qmod(tdly+1:end); %dus pas vanaf dat i = 1 voor het eerst effect op 
 %output flow ((bv.) 7 uur later = tdly)
 Qmod=[Qmod;NaN*ones(rem(length(Qmod),24),1)]; %aanvullen met eentjes tot 
 %een veelvoud van 24 (zo is reshape mogelijk)
 Qmod=reshape(Qmod,24,length(Qmod)/24); %naar matrix met 24 rijen (één voor
 %elk uur, kolomgewijs aangevuld => elke kolom is een dag)
 Qmod=nanmean(Qmod)'; %nanmean behandelt NaN als missing value en neemt gem
 %zonder deze waarde, neemt dit gemiddelde per kolom (dus per dag)
 % ' => transponeren => output = kolomvetor met debiet PER DAG uitgemiddeld

% %Qmod bevat het gemodelleerde debiet in m3/s
% 
% %% Output
% % kleine wijziging door Pieter, Q-kolom (3) in mm ipv m3/s
% Qmm=Qmod.*1000.*3600./Am2;
% PDMres=[T P Qmm S1./Smax];
