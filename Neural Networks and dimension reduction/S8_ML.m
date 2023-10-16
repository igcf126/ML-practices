clear 
clc
tic

data = readtable('auto-mpg.csv');
dataset = data(randperm(length(data.mpg)), :); % "Dataset" porque son los datos con los que trabajaremos

i = round(length(dataset.mpg)*0.75);

traindata = dataset(1:i, :);
testdata = dataset((i+1):end, :);

% dataset.mpg(:) == [traindata.mpg(:); testdata.mpg(:)]
i = length(traindata.cylinders);
D1 = ones(i, 1);

X = [ones(length(traindata.cylinders), 1) traindata.mpg traindata.displacement traindata.weight traindata.acceleration traindata.model_year traindata.origin]; % traindata.horsepower de tercero
r = traindata.cylinders;

w = inv(X'*X)*X'*r;

X2 = [ones(length(testdata.cylinders),1) testdata.mpg testdata.displacement testdata.weight testdata.acceleration testdata.model_year testdata.origin]; % testdata.horsepower de tercero

cylinderML = round(X2*w);
e = [cylinderML testdata.cylinders abs(cylinderML-testdata.cylinders)./testdata.cylinders*100]; 
emean = mean(e(:,3))

%%

X = [traindata.mpg traindata.displacement traindata.weight traindata.acceleration traindata.model_year traindata.origin]; % traindata.horsepower de tercero
X2 = [testdata.mpg testdata.displacement testdata.weight testdata.acceleration testdata.model_year testdata.origin]; % testdata.horsepower de tercero
r = traindata.cylinders;

k=3;
sel = zeros(length(X(1,:)), 1);
% for k = 1:length(X(1,:))
l=1;
b=0;
a=1;
c=0;
while a~=b
    c = c + 1;
    a=b;
    sel = zeros(length(X(1,:)), 1);
    for j = 1:length(X(1,:))
        if l==j
            continue
        else
            for i = 1:length(testdata.mpg)
                i;
                d = ((X2(i,[l j]) - X(:,[l j])).^2);
                d = sqrt(d(:,1)+d(:,2));
                
                [dis, pos] = mink(d, k);
                cylinderML(i,1) = mode(traindata.cylinders(pos));
                kNNi = sum(cylinderML(1)==testdata.cylinders(:))/length(testdata.cylinders);
            end
        end
        sel(j) = kNNi;
    end
    l = find(sel==max(sel));
    l = min(l);
    b=l;
    if c==length(X(1,:))
        break
    end
end



kNNe = sum(cylinderML==testdata.cylinders(:))/length(testdata.cylinders)

%%

X = [traindata.mpg traindata.displacement traindata.weight traindata.acceleration traindata.model_year traindata.origin]; % traindata.horsepower de tercero
X2 = [testdata.mpg testdata.displacement testdata.weight testdata.acceleration testdata.model_year testdata.origin]; % testdata.horsepower de tercero
r = traindata.cylinders;

m = mean(X);
[V D] = eig(cov(X));
[M I] = maxk(sum(D), 2);
Wpca = V(:,I);

Zpca = X*Wpca;
Z2pca = X2*Wpca;

%%
k=3;
for i = 1:length(testdata.mpg)
    i;
    d = ((Z2pca(i,:) - Zpca).^2);
    d = sqrt(d(:,1)+d(:,2));
    
    [dis, pos] = mink(d, k);
    cylinderML2(i,1) = mode(traindata.cylinders(pos));
end

kNNpca = sum(cylinderML2==testdata.cylinders(:))/length(testdata.cylinders)

%%
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
[M I] = maxk(sum(D), 2);
Wlda = V(:,I);

Zlda = X*Wlda;
Z2lda = X2*Wlda;

%%
k=3;
for i = 1:length(testdata.mpg)
    i;
    d = ((Z2lda(i,:) - Zlda).^2);
    d = sqrt(d(:,1)+d(:,2));
    
    [dis, pos] = mink(d, k);
    cylinderML3(i,1) = mode(traindata.cylinders(pos));
end

kNNlda = sum(cylinderML3==testdata.cylinders(:))/length(testdata.cylinders)


toc


%% 
% Métrica de error de diferencias, plotear, guardar errores y ¿comparar con
% variosa k?