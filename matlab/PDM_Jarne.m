function [Qmod] = PDM_Jarne(inputs,X) 
% PDM model gebaseerd op het artikel van Pieter Cabus (oorspronkelijk uit
% Moore 2007). Gebruik maken van Pareto distributie om bodemreservoir te
% beschrijven
% 
% Qmod: voorspeld debiet
% inputs: neerslag P, evaporatie Ep en oppervlakte A
% X: array van de parameters;


 P = inputs.P; % uurlijkse neerslag [mm/u]; 
 Ep = inputs.Ep; % uurlijkse evaporatie [mm/u];
 A = inputs.A; % Oppervlakte van het stroomgebied [km2]


%% definitie matrixdimensies (initialiseren van de toestandmatrices)
uren = length(P);

Ea = zeros(uren,1); % Actuele evaporatie [mm]
Qd = zeros(uren,1); % Directe runoff (richting oppervlakte opslagreservoir) [mm/u]
D = zeros(uren,1); % Drainage (richting grondwater opslagreservoir) [mm/u]
C = zeros(uren,1); % Critical Storage capaciteit in bodemreservoir [mm]
S1 = zeros(uren,1); % Opslag in bodemreservoir onder surface tension [mm]
pi = zeros(uren,1); % Netto neerslagoverschot [mm]
Qb = zeros(uren,1); % Basisafvoer (uit grondwater opslagreservoir) [mm/u]
Qbm3s = zeros(uren,1); % Basisafvoer (uit grondwater opslagreservoir) [m3/s]
% of1 = zeros(uren,1); % [mm/u] (NIET GEBRUIKT)
% of1m3s = zeros(uren,1); % [m3/s] (NIET GEBRUIKT)
% of2 = zeros(uren,1); % [mm/u] (NIET GEBRUIKT)
% of2m3s = zeros(uren,1); % [m3/s] (NIET GEBRUIKT)
Qmod = zeros(uren,1); % Voorspelde totale debiet (output van ons model) [m3/s]

% Extra Pieter
Sb = zeros(uren,1); % Opslag in grondwater opslagreservoir [mm]
qr = zeros(uren,1); % Oppervlakte afvoer (uit oppervlakte opslagreservoir) [mm/u]
qrm3s = zeros(uren,1); % Oppervlakte afvoer (uit oppervlakte opslagreservoir [m3/s]

%% Load parameters (X)
cmax = X(1); % Maximum storage capaciteit in bodemreservoir [mm]
cmin = X(2); % Minimum storage capaciteit in bodemreservoir [mm]
b = X(3); % Exponent voor de Pareto-distributie [-]
be = X(4); % Constante bij verhouding actueel/potentieel evaporatie [-] (= 1 bij lineaire vorm, = 2 bij quadratische vorm)
k1 = X(5); % Tijdsconstante 1 bij de routing in oppervlakte opslagreservoir met 2 lineaire reservoirs [u]
k2 = X(6); % Tijdsconstante 2 bij de routing in oppervlakte opslagreservoir met 2 lineaire reservoirs [u]
kb = X(7); % Basisafvoer tijdsconstante [u/mm2]
kg = X(8); % Drainage tijdsconstante [u]  
St = X(9); % Opslagcapaciteit in bodemreservoir onder surface tension [mm]   
bg = X(10); % Exponent voor de drainage functie [-]
tdly = ceil(X(11)); % Time delay constante [u]
qconst = X(12); % Constante rest flow [m3/s]
rainfac = X(13); % Corrigeren voor ruimtelijk verdeelde neerslag? [-] #### VRAAG ####

Smax = ((b*cmin)+cmax)/(b+1); % Totale hoeveelheid opslag in bodemreservoir (gelijk aan c_streep) [mm]
deltat = 1; %tijdstap van 1 uur [u]
Am2 = A*1000*1000; % Omzetting oppervlakte stroomgebied van km2 naar m2 [m2]

% Extra initialisatie Pieter (implementatie 2 lineaire reservoirs)
% Gebruikt voor de oppervlakte opslagreservoir

delta1x = exp(-deltat/k1);
delta2x = exp(-deltat/k2);
delta1 = -(delta1x + delta2x);
delta2 = delta1x * delta2x;
% Wanneer k1 en k2 niet gelijk zijn aan elkaar:
omega0 = (k1*(delta1x-1) - k2*(delta2x-1))/(k2-k1);
omega1 = ((k2*(delta2x-1)*delta1x) - (k1*(delta1x-1)*delta2x))/(k2-k1);
Sb(1) = 0.001; % Initiële opslag in het grondwater reservoir [mm]

% ###################################################
% Begin Berekeningen
% ###################################################

start = 1;
i = start; %tijdstap 1

% Initiële wateropslag (schatting!)
X(14) = Smax/2;
S1(i) = X(14); % Initiële bodemwater opslag vastgehouden onder surface tension [mm] (+- toestand 0 -> ook gelijk aan toestand op tijdstip 1?)


%% Berekening C(1), Ea(1), D(1) en pi(1) voor de eerste tijdstap

% 1) Critical Storage Capacity C

C(i) = cmin+(cmax-cmin)*(1-((Smax-S1(i))/(Smax-cmin))^(1/b+1)); % Opmerking: Smax = c_streep
%C(i) = cmax*(1-(1-S1(i)/Smax)^(1/(b+1))); -> Bij enkel een cmax

if C(i) > cmax % Controle op de geldigheid van het resultaat
    C(i) = cmax;
elseif C(i) <= 0
    C(i) = 0;
end

% 2) Actual evaporatie Ea

Ea(i) = Ep(i,1)*(1-((Smax-S1(i))/Smax)^be);

% 3) Drainage flow D 

if S1(i) > St
    D(i) = (1/kg)*((S1(i)-St)^bg);
else
    D(i) = 0; % Indien water onder surface tension vastgehouden kan worden
end

% 4) Netto neerslagoverschot pi

pi(i)= P(i,1) - Ea(i) - D(i);

%% Berekeningen voor elke volgende tijdstap i

i = start+1;
imax = uren;

for i = i:imax

    % 1) Actuele evaporatie Ea:

    Ea(i) = Ep(i,1)*(1-((Smax-S1(i-1))/Smax)^be); % Berekenen actuele evaporatie met behulp van vorige waarde van bodemreservoir S1

    % 2) Drainage naar grondwaterreservoir D:

    if S1(i-1) > St
        D(i) = (1/kg)*((S1(i-1)-St)^bg);
    else
        D(i)=0; % Indien water onder surface tension vastgehouden kan worden
    end

    % 3) Netto neerslagoverschot pi:

    pi(i) = P(i,1) - Ea(i) - D(i);

    % 4) Directe runoff Qd:

    voorwaarde = 0;
    % volstaat de voorwaarde P(i,1)>0 (Bruno) of moet de nettoneerslag ook
    % groter zijn dan 0
    %if (P(i,1)>0)

    if (pi(i)>0) % Eerste voorwaarde = Netto neerslag > 0
        voorwaarde = voorwaarde+1;
    end

    if (C(i-1)>cmin) % Tweede voorwaarde = Critical storage capacity > cmin -> enkel dan zou runoff theoretisch kunnen optreden
        voorwaarde = voorwaarde+1;
    end

    % Als beide voorwaarden zijn voldaan dan :
    if voorwaarde == 2
        %Qd(i)=(pi(i))*(1-((1-((C(i-1)-cmin)/(cmax-cmin)))^b));% hoe komt hij aan deze vergelijking ?
        % Pieter en Els deden een poging de integraal op te lossen (vgl.6
        % +vgl 5 (Moore)
        
        % Zie uitwerking in schriftje!
        if (C(i-1) + pi(i))< cmax
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1)-((cmax-C(i-1)-pi(i))/(cmax-cmin))^(b+1));
            
        else % Als de nettoneerslag pi zorgt voor een volledige verzadiging van bodemreservoir:
            Qd(i) = pi(i) - ((cmax - cmin)/(b+1))*(((cmax-C(i-1))/(cmax-cmin))^(b+1))+(C(i-1)+pi(i)-cmax);
            % Laatste term is extra runoff die optreedt buiten de grenzen
            % van het probability distributed opslagreservoir en dus de
            % cmax
        end
    else
        Qd(i)=0; % Indien beide voorwaarden voor runoff productie niet voldaan zijn
    end


    if Qd(i) < 0 % Geldigheid van het resultaat nagaan
        Qd(i) = 0;
    end

    % ---------------------------------------------------------------
    % Aanpassing toestanden mbv voorgaande berekende fluxen
    % ---------------------------------------------------------------

    % 5) Opslag in onverzadigde zone (bodemreservoir) S1:

    S1(i) = S1(i-1) + pi(i) - Qd(i); %mag niet groter worden dan Smax

    if S1(i) > Smax % Controle op geldigheid S1
        S1(i) = Smax;
    elseif S1(i) < 0 % mag niet negatief worden
        S1(i) = 0;
    end

    % 6) Kritische Storage Capacity C:

    C(i) = cmin+((cmax-cmin)*(1-(((Smax-S1(i))/(Smax-cmin))^(1/(b+1))))); % Appendix Moore: vgl 5 Pareto Distributie

    if C(i) > cmax % Controle op geldigheid C
        C(i) = cmax;
    elseif C(i) <= 0
        C(i) = 0;
    end

    %basisafvoer (Qb in mm/u)
    %Qb(i)=(D(i)*deltat/(kb+0.5*deltat))+(Qb(i-1)*(kb-0.5*deltat)/(kb+0.5*deltat));
    %%cfr. RVL -> Alternatief?

    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%

    % 7) Basisafvoer Qb en opslag in grondwaterreservoir Sb: 

    Sb(i) = Sb(i-1) - (1/(3*kb*(Sb(i-1)^2)))*(exp(-3*kb*(Sb(i-1)^2*deltat))-1)*(D(i)-kb*(Sb(i-1)^3)); % Recursieve formule voor opslag grondwaterreservoir
    Qb(i) = kb*(Sb(i)^3); % Basisafvoer met Horton-Izzard vergelijking

    if Qb(i) < 0 % Controle op geldigheid Qb 
        Qb(i)=0;
    end

    Qbm3s(i) = Qb(i)/1000*Am2/3600; % Oorspronkelijke eenheid = mm/u of l/u m2:
    % /1000 -> Omzetting van l of dm3 naar m3
    % *Am2 -> Vermenigvuldigen met totale oppervlakte stroomgebied want nu
    % nog maar enkel balans toegepast voor 1 m2
    % /3600 -> Omzetting van u naar s


    %oppervlakkige afvoer (cfr RVL) (of2 in mm/u) -> Alternatief?
    %of1(i)=(Qd(i)*deltat/(k1+0.5*deltat))+(of1(i-1)*(k1-0.5*deltat)/(k1+0.5*deltat)); %cfr. RVL
    %of1m3s(i)=of1(i)/1000*Am2/3600;
    %of2(i)=(of1(i)/(k2+0.5*deltat))+(of2(i-1)*(k2-0.5*deltat)/(k2+0.5*deltat)); %cfr. RVL
    %of2m3s(i)=of2(i)/1000*Am2/3600;

    %%%%%%%%%%%%%%%%%
    % volgens publicatie Moore
    %%%%%%%%%%%%%%%%%

    % 8) Directe afvoer qr:

    if i > 2 % Pas starten bij derde stap want Vergelijking 26 uit Moore -> tot 2 stappen terug
        qr(i) = -delta1*qr(i-1) - delta2*qr(i-2) + omega0*Qd(i) + omega1*Qd(i-1); % Transfer functie
        qrm3s(i)=qr(i)/1000*Am2/3600; % Zelfde omzetting van mm/u naar m3/s zoals bij basisafvoer
    end

    % 9) Totale afvoer Qmod:

    Qmod(i+tdly,1)= Qbm3s (i) + qrm3s(i) + qconst; % Volledige afvoer + constante term toevoegen



    %     if mod(i,100000)==0
    %         Qmm=Qmod.*1000.*3600./Am2;
    %         PDMres=[T P D pi Ep Ea Qd Qb Qmm S1./Smax];
    %         save('tmp_PDM_output','PDMres');
    %     end

end

Qmod = Qmod(tdly+1:end); % Time delay zorgt voor latere effect op output
Qmod = [Qmod;NaN*ones(rem(length(Qmod),24),1)]; % Op einde van matrix aanvullen met NaN tot veelvoud van 24 -> nodig voor de reshape
Qmod = reshape(Qmod,24,length(Qmod)/24); % Dagen in de kolommen, uren in de rijen -> door aanvulling bevat laatste kolom (laatste dag) mogelijks NaN waarden!
Qmod = nanmean(Qmod)'; % Eerst berekenen van mean per kolom (of dus dag) zonder NaN, daarna transponeren zodat we kolomvector verkrijgen

% %Qmod bevat het gemodelleerde debiet in m3/s
% 
% %% Output
% % kleine wijziging door Pieter, Q-kolom (3) in mm ipv m3/s
% Qmm=Qmod.*1000.*3600./Am2;
% PDMres=[T P Qmm S1./Smax];