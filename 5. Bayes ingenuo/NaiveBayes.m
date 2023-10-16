%% Naives Ingenuo

tic
clear
clc

% Se leen los datos del csv y se aleatoriza para prevenir cualquier
% influencia por un orden que se desconozca.

data = readtable("train.csv"); % "Data" porque son los datos que nos dan
traindata = data(randperm(length(data.Name)), :); % "Dataset" porque son los datos con los que trabajaremos

testdata = readtable("test.csv");
tester = readtable("gender_submission.csv");

vivos = traindata(traindata.Survived==1, :);
muertos = traindata(traindata.Survived==0, :);

clear data

%% Conocemos el teorema de Bayes, que trata sobre las probabilidades a priori.

PVivo = sum(traindata.Survived==1)/length(traindata.Survived);
PMuerto = sum(traindata.Survived==0)/length(traindata.Survived);

% En los datos de entrenamiento tenemos el ID de pasajero, el nombre, cabina
% y el tícket, 
% que son características únicas por pasajero. Se ignora.

% Se selecciona la característica Pclass y Sex, porque en el montexto era
% importante en la toma de decisiones importantes. Pues a la clase de cada
% persona y su sexo se le daba importancia y distinción, incluyendo a la
% hora de seleccionar prioridades.

% La edad se ignora por ser un valor continuo y se trabajará con la frecuencia de los valores
% Embarked se ignora porque el origen de las personas parece no importar
% del todo.

% El valor de Cabin será ignorado por ser muy variable y a la vez por
% tener poca información en sí.

% Finalmente, se toma la cantidad de hermanos y familiares, SibSp, Parch

%% Survived, Pclass, SibSp, Parch.
% sex, 

liveMen = sum(vivos.SexN==0)/length(vivos.SexN);
liveWomen = sum(vivos.SexN==1)/length(vivos.SexN);

deathMen = sum(muertos.SexN==0)/length(muertos.SexN);
deathWomen = sum(muertos.SexN==1)/length(muertos.SexN);

[vivo_freqSex vivo_NSex] = hist(vivos.SexN, min(vivos.SexN):1:max(vivos.SexN));
vivo_freqSex = vivo_freqSex./length(vivos.SexN);

[muerto_freqSex muerto_NSex] = hist(muertos.SexN, min(muertos.SexN):1:max(muertos.SexN));
muerto_freqSex = muerto_freqSex./length(muertos.SexN);


%% Histogramas para gente viva

[vivo_freqPclass vivo_NPclass] = hist(vivos.Pclass, min(vivos.Pclass):1:max(vivos.Pclass));
[vivo_freqSibSp vivo_NSibSp] = hist(vivos.SibSp, min(vivos.SibSp):1:max(vivos.SibSp)*2);
[vivo_freqParch vivo_NParch] = hist(vivos.Parch, min(vivos.Parch):1:max(vivos.Parch)*3);

vivo_freqPclass = vivo_freqPclass./length(vivos.Pclass);
vivo_freqSibSp = vivo_freqSibSp./length(vivos.SibSp);
vivo_freqParch = vivo_freqParch./length(vivos.Parch);

%% Histogramas para gente viva

[muerto_freqPclass muerto_NPclass] = hist(muertos.Pclass, min(muertos.Pclass):1:max(muertos.Pclass));
[muerto_freqSibSp muerto_NSibSp] = hist(muertos.SibSp, min(muertos.SibSp):1:max(muertos.SibSp));
[muerto_freqParch muerto_NParch] = hist(muertos.Parch, min(muertos.Parch):1:max(muertos.Parch)*3);

muerto_freqPclass = muerto_freqPclass./length(muertos.Pclass);
muerto_freqSibSp = muerto_freqSibSp./length(muertos.SibSp);
muerto_freqParch = muerto_freqParch./length(muertos.Parch);


%%
%% Likelihood por sexo
minitest = testdata(:,:);
ulu = mod(find(vivos.SexN(:)==minitest.SexN'),length(traindata.PassengerId))+1;
% ulu = reshape(ulu, length(traindata.PassengerId), [])

i = vivo_NSex(:)==minitest.SexN';
i = mod(find(i)-1, length(vivo_NSex))+1;
likeVivoSex = vivo_freqSex(i);

i = muerto_NSex(:)==minitest.SexN';
i = mod(find(i)-1, length(muerto_NSex))+1;
likeMuertoSex = muerto_freqSex(i);

%% Likelihood por Pclass

i = vivo_NPclass(:)==minitest.Pclass';
i = mod(find(i)-1, length(vivo_NPclass))+1;
likeVivoPclass = vivo_freqPclass(i);

i = muerto_NPclass(:)==minitest.Pclass';
i = mod(find(i)-1, length(muerto_NPclass))+1;
likeMuertoPclass = muerto_freqPclass(i);

%% Likelihood por SibSp

i = vivo_NSibSp(:)==minitest.SibSp';
i = mod(find(i)-1, length(vivo_NSibSp))+1;
likeVivoSibSp = vivo_freqSibSp(i);

i = muerto_NSibSp(:)==minitest.SibSp';
i = mod(find(i)-1, length(muerto_NSibSp))+1;
likeMuertoSibSp = muerto_freqSibSp(i);

%% Likelihood por Parch

i = vivo_NParch(:)==minitest.Parch';
i = mod(find(i)-1, length(vivo_NParch))+1;
likeVivoParch = vivo_freqParch(i);

i = muerto_NParch(:)==minitest.Parch';
i = mod(find(i)-1, length(muerto_NParch))+1;
likeMuertoParch = muerto_freqParch(i);


gVivo = PVivo.*likeVivoSex.*likeVivoPclass.*likeVivoSibSp.*likeVivoParch;
gMuerto = PMuerto.*likeMuertoSex.*likeMuertoPclass.*likeMuertoSibSp.*likeMuertoParch;

MLSurvived = nan(length(minitest.PassengerId),1);
minitest = [minitest table(MLSurvived)];

minitest.MLSurvived(:) = gVivo > gMuerto;
NaiveAccuracy = sum(minitest.MLSurvived==tester.Survived(:))/length(tester.Survived)

ind = tester.Survived(find(minitest.MLSurvived~=tester.Survived(:)));
falseNegative = sum(ind)/length(tester.Survived)
falsePositive = sum(~ind)/length(tester.Survived)

toc

%% 
% Si observamos los casos en que no se ha acertado el valor de
% supervivencia podemos observar que se presentan outliers, por lo que
% puede ser considerada una razón por la que tenemos fallos en los
% aciertos.

minitest(find(minitest.MLSurvived~=tester.Survived(:)), :);

toc

%%
tic
MLSurvived = nan(length(testdata.PassengerId),1);
testdata = [testdata table(MLSurvived)];

for k = 1:2:7
    for i = 1:length(testdata.PassengerId)
        i;
        d = (([testdata.SexN(i), testdata.Pclass(i), testdata.SibSp(i), testdata.Parch(i)] - [traindata.SexN(:), traindata.Pclass(:), traindata.SibSp(:), traindata.Parch(:)]).^2);
        d = sqrt(d(:,1)+d(:,2)+d(:,3)+d(:,4));
    
        [dis, pos] = mink(d, k);
        testdata.MLSurvived(i) = mode(traindata.Survived(pos));
    end
    k
    kNN = sum(testdata.MLSurvived==tester.Survived(:))/length(tester.Survived)
end

toc