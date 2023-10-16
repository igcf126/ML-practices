clear
clc
tic

data = readtable('auto-mpg.csv');
dataset = data(randperm(length(data.mpg)), :); % "Dataset" porque son los datos con los que trabajaremos

i = round(length(dataset.mpg)*0.75);

traindata = dataset(1:i, :);
testdata = dataset((i+1):end, :);

i = length(dataset.cylinders);
D1 = ones(i, 1);

X = [traindata.mpg traindata.displacement traindata.weight traindata.acceleration traindata.model_year traindata.origin]; % traindata.horsepower de tercero
X2 = [testdata.mpg testdata.displacement testdata.weight testdata.acceleration testdata.model_year testdata.origin]; % testdata.horsepower de tercero
r = traindata.cylinders;

%% Implementación de LDA

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
    Si = i*(X-mi)'*(X-mi);
    Sw = Sw + Si;
    i;
    
    mcent(ind, :) = mi;  
end

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

% Des = sum(D, 'all');
% cant = 0;
% posi = sum(D);
% Eigenes = 0;
% while Eigenes<=0.9 %% Ch6, Diapo 11
%     cant = cant + 1;
%     Eigenes = sum(posi(1:cant))/Des;
% end
% cant

cant = length(find(D>max(D, [], 'all')/1000)) %% El diez es la división al valor máximo de la matriz para asumir los menores a max/10 como cero.
[M I] = maxk(sum(D), cant);
Wlda = V(:,I)

Zlda = X*Wlda;
Z2lda = X2*Wlda;

% w = inv(Zlda'*Zlda)*Zlda'*r;
% 
% cylinderML = round(Zlda*w);
% e = [cylinderML dataset.cylinders cylinderML==dataset.cylinders]; 
% emean = sum(e(:,end))/length(cylinderML)


k=1;
for i = 1:length(testdata.mpg)
    i;
    d = ((Z2lda(i,:) - Zlda).^2);
    d = sqrt(d(:,1)+d(:,2));
    
    [dis, pos] = mink(d, k);
    cylinderML3(i,1) = mode(traindata.cylinders(pos));
end

kNNlda = sum(cylinderML3==testdata.cylinders(:))/length(testdata.cylinders)

toc