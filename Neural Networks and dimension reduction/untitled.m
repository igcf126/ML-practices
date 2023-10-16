clear 
clc

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


[fcyl Ncyl] = hist(traindata.cylinders, min(traindata.cylinders):1:max(traindata.cylinders));
% Si = 0;
Sw = 0;
Sb = 0;
[a, b] = size(X);
mcent = nan(length(Ncyl(fcyl~=0)), b);

for i = Ncyl(fcyl~=0)
    ind = find(Ncyl==i);
    Xi = X(traindata.cylinders(:)==i,:);
    mi = mean(Xi);
    Si = (X-mi)'*(X-mi);
    Sw = Sw + Si;
    i;
    
    mcent(ind, :) = mi;  
end
Sw

mt = mean(mcent, 'omitnan');

for i = Ncyl(fcyl~=0)
    ind = find(Ncyl==i);

    Xi = X(traindata.cylinders(:)==i,:);
    mi = mean(Xi);

    Sj = fcyl(ind)*(mi-mt)'*(mi-mt);
    Sb = Sb + Sj;
end

Sw;
Sb;

[V D] = eig(inv(Sw)*Sb);

cant = length(find(D>max(D, [], 'all')/1000)) %% El diez es la división al valor máximo de la matriz para asumir los menores a max/10 como cero.
[M I] = maxk(sum(D), cant);
Wlda = V(:,I)


Zlda = [X*Wlda];
Z2lda = [X2*Wlda];

%% (a-b)²=a²+b²-2ab 
% sum(X.^2, 2) = a² => repmat(a2, 1, length(sum(X2'.^2, 1) ))
% sum(X2'.^2, 1) = b^2 => repmat(b2, length(sum(X.^2, 2)), 1)
% -2*X*X2'

a = sum(X.^2, 2);
b = sum(X2'.^2, 1);
% a2 = repmat(a, 1, length(b))
% b2 = repmat(b, length(a), 1)

ab_2 = -2*X*X2'

prodkNN = sqrt(a+b+ab_2) %sqrt(Zlda*Z2lda')

[value pos] = mink(prodkNN, 3)

for i = 1:length(r2)
    cylinderML(i,1) = mode(r(pos(:, i)));
end

[cylinderML r2]

aciertosLDA = sum(cylinderML==r2)/length(r2)

