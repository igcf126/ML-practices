clc
clear
tic
%%
X = 1:1000;
X = X';
w = [-0.7 220];
Y = sum ([ones(length(X), 1) X].*repmat(w, length(X), 1), 2);
Y = Y + normrnd(0, 11, length(X), 1);

data = [X Y];

dataset = data(randperm(length(X)), :); % "Dataset" porque son los datos con los que trabajaremos

i = round(length(X)*0.75);
D1 = ones(i, 1);

traindata = dataset(1:i, :);
testdata = dataset((i+1):end, :);

X_train = [ones(i,1) traindata(:,1)];
r = traindata(:, 2);

wA1 = inv(X_train'*X_train)*(X_train'*r)

X_test = [ones(length(testdata(:,1)),1) testdata(:,1)];
YML = X_test*wA1;

e1 = [YML testdata(:,2) (YML-testdata(:,2)).^2]; 
emean1 = mean(e1(:,3))

%%
data2 = readtable('auto-mpg.csv');
dataset2 = data2(randperm(length(data2.mpg)), :); 
% "Dataset" porque son los datos con los que trabajaremos

i = round(length(dataset2.mpg)*0.75);

traindata2 = dataset2(1:i, :);
testdata2 = dataset2((i+1):end, :);

i = length(traindata2.cylinders);
D2 = ones(i, 1);

X2 = [ones(length(testdata2.cylinders(:)),1) testdata2.cylinders testdata2.displacement testdata2.weight testdata2.acceleration testdata2.model_year testdata2.origin]; 
X = [ones(i,1)  traindata2.cylinders traindata2.displacement traindata2.weight traindata2.acceleration traindata2.model_year traindata2.origin]; 
r = traindata2.mpg;

sel = zeros(length(X(1,:)), 1);
l=2;
b=2;
a=0;
c=0;
while a~=b
    c = c + 1;  
    a=b;
    sel = zeros(length(X(1,:)), 1);
    for j = 2:length(X(1,:))
        if a==j
            continue
        else
            wA2 = inv(X(:, [1 a j])'*X(:, [1 a j]))*X(:, [1 a j])'*r;

            YML = X2(:, [1 a j])*wA2;
            
            e2 = [YML testdata2.mpg (YML-testdata2.mpg).^2 1-abs(YML-testdata2.mpg)./length(testdata2.mpg)]; 
            emean = mean(e2(:,3));
            eprop = mean(e2(:,4));
        end
        sel(j) = eprop;
    end
    l = find(sel==max(sel));
    l = min(l);
    b=l;
    if c==length(X(1,:))
        break
    end
end

% l
% j
% a
% b

wA2 = inv(X(:, [1 a b])'*X(:, [1 a b]))*X(:, [1 a b])'*r

YML = X2(:, [1 a b])*wA2;

e2 = [YML testdata2.mpg (YML-testdata2.mpg).^2 1-abs(YML-testdata2.mpg)./length(testdata2.mpg)]; 
emean2a = mean(e2(:,3))
eprop = mean(e2(:,4));

%% 

traindata2 = dataset2(1:i, :);
testdata2 = dataset2((i+1):end, :);

i = length(traindata2.cylinders);
j = length(testdata2.cylinders);

X2 = [ones(j,1) testdata2.cylinders testdata2.displacement testdata2.weight testdata2.acceleration testdata2.model_year testdata2.origin]; % testdata.horsepower de tercero
X = [ones(i,1)  traindata2.cylinders traindata2.displacement traindata2.weight traindata2.acceleration traindata2.model_year traindata2.origin]; % traindata.horsepower de tercero
r = traindata2.mpg;

for d = 2:length(X(1,:))
    disp('Pesos y error para polinomio de grado ') 
    disp(d-1)

    wA3 = inv(X(:,1:d)'*X(:,1:d))*X(:,1:d)'*r
    mpgML = X2(:,1:d)*wA3;

    e3 = [mpgML testdata2.mpg (mpgML-testdata2.mpg).^2]; 
    emean3 = mean(e3(:,3))
end


