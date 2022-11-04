%% ##########################################
% INLADEN GEGEVENS
% ###########################################
clear;
clc;

%% Meteorologische data inlezen
% -------------------------------------------------------------
data_neerslag = readtable("MaarkeKerkem_Neerslag_1h_geaccumuleerd.csv"); 
data_evapo = readtable("Liedekerke_ME_Potential_evapotranspiration_1h_geaccumuleerd.csv");
data_debiet = readtable("OS266_L06_342_Afvoer_hourly_reprocessed.csv");
% data_debiet2 = readtable("OS266_LS06_34A_Afvoer_hourly_reprocessed.csv");
% -> NIET GEBRUIKT (vermoedelijk basisafvoer?)
% -------------------------------------------------------------

%% Tijd selecteren waarbij data overlapt
% -------------------------------------------------------------
data_begin =  [data_neerslag.Datum(1),data_evapo.Date(1),data_debiet.Date(1)];
begin_datum = datenum(max(data_begin));
data_eind =  [data_neerslag.Datum(end),data_evapo.Date(end),data_debiet.Date(end)];
eind_datum = datenum(min(data_eind));

% Neerslag: Indien NaN => vervangen door 0
% -----------------------------
id_neerslag = datenum(data_neerslag.Datum) >= begin_datum & datenum(data_neerslag.Datum) <= eind_datum;
data_neerslag = data_neerslag(id_neerslag,:);
for i = 1:size(data_neerslag,1)
    if isnan(data_neerslag.Neerslag(i))
        data_neerslag.Neerslag(i) = 0; % NaN waarden veranderen naar 0 
    end
end
% ----------------------------

% Evaporatie: Indien NaN of negatief => vervangen door 0
% ---------------------------
id_evapo = datenum(data_evapo.Date) >= begin_datum & datenum(data_evapo.Date) <= eind_datum;
data_evapo = data_evapo(id_evapo,:);
for i = 1:size(data_evapo,1)
    if isnan(data_evapo.ET(i)) || data_evapo.ET(i) < 0 
        data_evapo.ET(i) = 0; % NaN waarden en negatieve waarden veranderen naar 0
    end
end
% ---------------------------

% Debiet: Geen aanpassing nodig 
% ---------------------------
id_debiet = datenum(data_debiet.Date) >= begin_datum & datenum(data_debiet.Date) <= eind_datum;
data_debiet = data_debiet(id_debiet,:);
% -------------------------------------

% -----------------------------------------------------------------

%% Data samenvoegen
% ----------------------------------------
Data = table(data_neerslag.Datum,data_neerslag.Neerslag,data_evapo.ET,data_debiet.Flow);
Data.Properties.VariableNames = {'Datum','Neerslag','Evaporatie','Debiet'};

% Enkel volledige dagen gebruiken! -> starten bij 00:00:00 en eindigen bij
% 23:00:00
id_first = find(Data.Datum.Hour == 00,1,"first");
id_last = find(Data.Datum.Hour == 23,1,"last");
Data = Data(id_first:id_last,:);

% -----------------------------------------
%% Oplossen van dubbele tijdstippen en overgeslagen uren
% -----------------------------------------
t = size(Data,1); % Totale lengte van de periode
[unieke_tijden, unieke_id] = unique(Data.Datum); % Enkel unieke tijdstippen selecteren
Data = Data(unieke_id,:); % Aanpassing dubbele aanwezigheid

time_tab = table2timetable(Data); % Omzetting naar timetable formaat
time_tab = retime(time_tab,'hourly'); % Zorgen dat de stapgrootte in tijd altijd één uur is
Data = timetable2table(time_tab); % Aanpassing uren overslaan

id_nan = find(isnan(Data.Evaporatie)); % id's van de extra toegevoegde NaN's door uren toe te voegen 
value_voor = [Data.Neerslag(id_nan - 1), Data.Evaporatie(id_nan - 1)];
value_na = [Data.Neerslag(id_nan + 1), Data.Evaporatie(id_nan + 1)];
Data.Neerslag(id_nan) = (value_voor(:,1) + value_na(:,1))./2; % Lineaire interpolatie voor neerslag
Data.Evaporatie(id_nan) = (value_voor(:,2) + value_na(:,2))./2; % Lineaire interpolatie voor evaporatie

% Data bevat nu geen dubbele data telkens met een uur verschil startend
% bij 00:00 en eindigen bij 23:00 met geen NaN in neerslag en evaporatie
% (ofwel NaN vervangen door 0; ofwel NaN vervangen
% door geïnterpoleerde waarde)

% -----------------------------------------

%% Range van parameters in PDM definiëren volgens Pieter Cabus (2008)
% ----------------------------------------------------------------------
% param =   [cmax cmin b   be k1  k2  kb   kg    St  bg tdly qconst]
param_min = [160  0    0.1 1  0.9 0.1 0    700   0   1  0   -4.08]; % Minimale parameterwaarden
param_max = [5000 300  2   2  40  15  5000 25000 150 2  24   0.03]; % Maximale parameterwaarden
param_namen = ["cmax","cmin","b","be","k1","k2","kb","kg","St","bg","tdly","qconst"]; % Parameternamen
p = length(param_max); % Aantal parameters

% OPM: Geen rainfactor gebruikt en parameter m is als sowieso niet nodig in
% PDM functie van Cabus
% Dit brengt totaal AANTAL PARAMETERS op 12

% -----------------------------------------------------------------------

%% Oppervlakte stroomgebied
% -----------------------------------------------------------------------
oppervlakte = 144; % Oppervlakte van Zwalm stroomgebied [km2] (bron: https://nl.wikipedia.org/wiki/Zwalm_(rivier))
% Eventueel aanpassen als een ander stroomgebied bekeken moet worden

% -----------------------------------------------------------------------

%% Opdeling data in een Kalibratie en Validatie set
% ------------------------------------------------------------------
% Opmerking: Data tussen 2004 en 2010 bevat veel NaN-waarden!

% Kalibratie set
% -----------------------------------
start_jaar = 2010;
eind_jaar = 2016;
Data_1016 = Data(Data.Datum.Year >= start_jaar & Data.Datum.Year <= eind_jaar,:); % Kalibratie dataset
t_1016 = size(Data_1016,1);
dagen = datetime(Data_1016.Datum,"Format","dd-MMM-uuuu")';
dagen_1016 = dagen(1):dagen(end); % Datetimes aanmaken
n_1016 = size(dagen_1016,2);
Q_1016 = reshape(Data_1016.Debiet,[24,n_1016]); % Observaties klaarmaken voor plot (gemiddelden per dag)
Q_1016 = mean(Q_1016,"omitnan")'; % Gemiddelden per dag berekenen
% -----------------------------------

% Validatie set
% -----------------------------------
start_jaar = 2017;
eind_jaar = 2021;
Data_1721 = Data(Data.Datum.Year >= start_jaar & Data.Datum.Year <= eind_jaar,:); % Validatie dataset
t_1721 = size(Data_1721,1);
dagen = datetime(Data_1721.Datum,"Format","dd-MMM-uuuu")';
dagen_1721 = dagen(1):dagen(end); % Datetimes aanmaken
n_1721 = size(dagen_1721,2);
Q_1721 = reshape(Data_1721.Debiet,[24,n_1721]); % Observaties klaarmaken voor plot (gemiddelden per dag)
Q_1721 = mean(Q_1721,"omitnan")'; % Gemiddelden per dag berekenen
% -----------------------------------

% ------------------------------------------------------------------
