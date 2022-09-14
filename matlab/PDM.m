 function [Qmod]=PDMPieter(inputs,X)

 P=inputs.P;
 Ep=inputs.Ep;
 A=inputs.A;
 
%P: hourly rainfall [mm/h];
%Ep: hourly evapotranspiration [mm/h];
%X: array of parameters;

%% definitie matrixdimensies
uren=length(P);

Ea      =zeros(uren,1); %mm
Qd      =zeros(uren,1); %mm/u
D       =zeros(uren,1); %mm/u
C       =zeros(uren,1); %mm
S1      =zeros(uren,1); %mm
pi      =zeros(uren,1); %mm
Qb      =zeros(uren,1); %mm/u
Qbm3s   =zeros(uren,1); %m3/s
of1     =zeros(uren,1); %mm/u
of1m3s  =zeros(uren,1); %m3/s
of2     =zeros(uren,1); %mm/u
of2m3s  =zeros(uren,1); %m3/s
Qmod    =zeros(uren,1); %m3/s

% extra Pieter
Sb      = zeros(uren,1); %mm
qr      = zeros(uren,1); %mm/u

%% Load parameters
cmax=X(1);
cmin=X(2);
b=X(3);
be=X(4);
k1=X(5);
k2=X(6);
kb=X(7);
kg=X(8);   
St=X(9);     
bg=X(10);
tdly=ceil(X(11));
qconst=X(12);
rainfac=X(13);

Smax=((b*cmin)+cmax)/(b+1);
deltat=1;               %tijdstap van 1 uur
Am2=A*1000*1000;

% extra initialisatie Pieter (implementatie 2 lineaire stores)
delta1x = exp(-1/k1);
delta2x = exp(-1/k2);
delta1  =-(delta1x + delta2x);
delta2  = delta1x * delta2x;
omega0  = (k1*(delta1x-1)-k2*(delta2x-1))/(k2-k1);
omega1  = ((k2*(delta2x-1)*delta1x)-(k1*(delta1x-1)*delta2x))/(k2-k1);
Sb(1)   = 0.001;

start=1;
i=start;                    %tijdstap 1

%initi�le wateropslag (schatting!)
X(14)=Smax/2;
S1(i)=X(14);


%berekening C*(1), Ea(1),D(1) en pi(1) voor de eerste tijdstap
C(i)=cmin+(cmax-cmin)*(1-((Smax-S1(i))/(Smax-cmin))^(1/b+1)); %cmax*(1-(1-S1(i)/Smax)^(1/(b+1)));
if C(i)>cmax
    C(i)=cmax;
elseif C(i)<=0
    C(i)=0;
end;

Ea(i)=Ep(i,1)*(1-((Smax-S1(i))/Smax)^be);
if S1(i)>St
        D(i)=(1/kg)*((S1(i)-St)^bg);
else
        D(i)=0;
end;
pi(i)=P(i,1)-Ea(i)-D(i);

%berekeningen voor elke volgende tijdstap i
i=start+1;
imax=uren;
for i=i:imax
    %actuele evapotranspiratie
    Ea(i)=Ep(i,1)*(1-((Smax-S1(i-1))/Smax)^be);
    %drainage naar grondwaterreservoir
    if S1(i-1)>St
        D(i)=(1/kg)*((S1(i-1)-St)^bg);
    else
        D(i)=0;
    end;
    %netto neerslagoverschot
    pi(i)=P(i,1)-Ea(i)-D(i);
    %directe runoff Qd
    voorwaarde=0;
    % volstaat de voorwaarde P(i,1)>0 (Bruno) of moet de nettoneerslag ook
    % groter zijn dan 0
    %if (P(i,1)>0)
    if (pi(i)>0)
        voorwaarde=voorwaarde+1;
    end;
    if (C(i-1)>cmin)
        voorwaarde=voorwaarde+1;
    end;
    if voorwaarde==2
        %Qd(i)=(pi(i))*(1-((1-((C(i-1)-cmin)/(cmax-cmin)))^b));% hoe komt hij aan deze vergelijking ?
        % Pieter en Els deden een poging de integraal op te lossen (vgl.6
        % +vgl 5 (Moore)
        if (C(i-1)+pi(i))< cmax
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1)-((cmax-C(i-1)-pi(i))/(cmax-cmin))^(b+1));
        % als de nettoneerslag zorgt voor een volledige verzadiging ...
        else
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1))+(C(i-1)+pi(i)-cmax);
        end
    else
        Qd(i)=0;
    end;
    if Qd(i)<0
        Qd(i)=0;
    end;
    %opslag in onverzadigde zone
    S1(i)=S1(i-1)+(pi(i))-Qd(i);      %mag niet groter worden dan Smax
    if S1(i)>Smax
        S1(i)=Smax;
    elseif S1(i)<0                      %mag niet negatief worden
        S1(i)=0;
    end;
    %C
    C(i)=cmin+((cmax-cmin)*(1-(((Smax-S1(i))/(Smax-cmin))^(1/(b+1)))));
    if C(i)>cmax
        C(i)=cmax;
    elseif C(i)<=0
        C(i)=0;
    end;
    %basisafvoer (Qb in mm/u)
    %Qb(i)=(D(i)*deltat/(kb+0.5*deltat))+(Qb(i-1)*(kb-0.5*deltat)/(kb+0.5*deltat)); %cfr. RVL
   
    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%
   
    Sb(i) = Sb(i-1) - (1/(3*kb*(Sb(i-1)^2)))*(exp(-3*kb*(Sb(i-1)^2))-1)*(D(i)-kb*(Sb(i-1)^3));
    Qb(i) = kb*(Sb(i)^3);
    if Qb(i)<0
        Qb(i)=0;
    end;
    Qbm3s(i)=Qb(i)/1000*Am2/3600;
    %oppervlakkige afvoer (cfr RVL) (of2 in mm/u)
    %of1(i)=(Qd(i)*deltat/(k1+0.5*deltat))+(of1(i-1)*(k1-0.5*deltat)/(k1+0.5*deltat)); %cfr. RVL
    %of1m3s(i)=of1(i)/1000*Am2/3600;
    %of2(i)=(of1(i)/(k2+0.5*deltat))+(of2(i-1)*(k2-0.5*deltat)/(k2+0.5*deltat)); %cfr. RVL
    %of2m3s(i)=of2(i)/1000*Am2/3600;
   
    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%
    if i > 2
    qr(i)=-delta1*qr(i-1)-delta2*qr(i-2)+omega0*Qd(i)+omega1*Qd(i-1);
    of2m3s(i)=qr(i)/1000*Am2/3600;
    end
   
    %totale afvoer
    Qmod(i+tdly,1)=Qbm3s(i)+of2m3s(i)+qconst;
    
   
   
%     if mod(i,100000)==0
%         Qmm=Qmod.*1000.*3600./Am2;
%         PDMres=[T P D pi Ep Ea Qd Qb Qmm S1./Smax];
%         save('tmp_PDM_output','PDMres');
%     end
   
end;

 Qmod=Qmod(tdly+1:end);
 Qmod=[Qmod;NaN*ones(rem(length(Qmod),24),1)];
 Qmod=reshape(Qmod,24,length(Qmod)/24);
 Qmod=nanmean(Qmod)';

% %Qmod bevat het gemodelleerde debiet in m3/s
% 
% %% Output
% % kleine wijziging door Pieter, Q-kolom (3) in mm ipv m3/s
% Qmm=Qmod.*1000.*3600./Am2;
% PDMres=[T P Qmm S1./Smax];