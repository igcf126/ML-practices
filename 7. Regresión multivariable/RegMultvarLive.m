clear
clc
tic
%% 
% *Práctica 7. Regresión multivariable.*
% 
% En el presente laboratorio se estará trabajando una regresión multivariable 
% tal y como se apreció en el laboratorio anterior. Trabajando con la misma base 
% de datos anterior. 
% 
% Primeramente, se hará una aleatorización y segmentación de los datos.

data = readtable('auto-mpg.csv');
dataset = data(randperm(length(data.mpg)), :); % "Dataset" porque son los datos con los que trabajaremos

dataset.horsepower = fillmissing(dataset.horsepower(:), 'constant', nanmean(dataset.horsepower));

i = round(length(dataset.mpg)*0.75);

traindata = dataset(1:i, :);
testdata = dataset((i+1):end, :);

i = length(dataset.cylinders);
D1 = ones(i, 1);

X = [traindata.mpg traindata.displacement traindata.horsepower traindata.weight traindata.acceleration traindata.model_year traindata.origin]; % traindata.horsepower de tercero
X2 = [testdata.mpg testdata.displacement testdata.horsepower testdata.weight testdata.acceleration testdata.model_year testdata.origin]; % testdata.horsepower de tercero
r = traindata.cylinders;
r2 = testdata.cylinders;
%% 
% A continuacón se computaron los pesos de la regresión lineal multivariable 
% utilizando el algoritmo ya conocido