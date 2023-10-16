clear
clc

% Se leen los datos del csv y se aleatoriza para prevenir cualquier
% influencia por un orden que se desconozca.

data = readtable("cardio_train.csv"); % "Data" porque son los datos que nos dan
dataset = data(randperm(length(data.id)), :); % "Dataset" porque son los datos con los que trabajaremos

% Se ha decidido por tomar un 70 % de los datos para el entrenamiento del
% programa, para as[i poner a prueba el otro 30 % del dataset.

% traindata = dataset(1:length(dataset.id)*0.7, :);
% testdata = dataset(1:length(dataset.id)*0.3, :);

i = round(length(dataset.id)*0.75);

traindata = dataset(1:i, :);
testdata = dataset((i+1):end, :);

clear data
clear dataset

%% Priors, características y likelihoods

% Para conocer el prior de los enfermos en el dataset, se extrae su
% proporción en base a los datos de prueba.

prior = [sum(traindata.cardio==0) sum(traindata.cardio==1)]./length(traindata.cardio);
PC1 = prior(2);

sanos = traindata(traindata.cardio==0, :);
enfermos = traindata(traindata.cardio==1, :);

likeSano = [mean(sanos.age) std(sanos.age); mean(sanos.height) std(sanos.height); mean(sanos.weight) std(sanos.weight)];
likeEnfermo = [mean(enfermos.age) std(enfermos.age); mean(enfermos.height) std(enfermos.height); mean(enfermos.weight) std(enfermos.weight)];

%% Gráfico de la edad de las personas enfermas versus su estimación
figure
edadEnfermoEstimado = normrnd(likeEnfermo(1, 1), likeEnfermo(1,2), [length(enfermos.id) 1]);
histogram(enfermos.age)
hold on
histogram(edadEnfermoEstimado)

title("Edad de las personas enfermas")
legend ("Datos reales", "Modelo estimado")
xlabel("Edad (días)")
ylabel("Cantidad de enfermos")

clear edadEnfermoEstimado

%% Gráfico de la altura de las personas enfermas versus su estimación
figure
alturaEnfermoEstimado = normrnd(likeEnfermo(2, 1), likeEnfermo(2,2), [length(enfermos.id) 1]);
histogram(enfermos.height)
hold on
histogram(alturaEnfermoEstimado)

title("Altura de las personas enfermas")
legend ("Datos reales", "Modelo estimado")
xlabel("Altura (cm)")
ylabel("Cantidad de enfermos")

clear alturaEnfermoEstimado

%% Gráfico del peso de las personas enfermas versus su estimación
figure
pesoEnfermoEstimado = normrnd(likeEnfermo(3, 1), likeEnfermo(3,2), [length(enfermos.id) 1]);
histogram(enfermos.weight)
hold on
histogram(pesoEnfermoEstimado)

title("Peso de las personas enfermas")
legend ("Datos reales", "Modelo estimado")
xlabel("Peso (kg)")
ylabel("Cantidad de enfermos")

clear pesoEnfermoEstimado

%%

%% Gráfico de la edad de las personas sanas versus su estimación
figure
edadSanoEstimado = normrnd(likeSano(1, 1), likeSano(1,2), [length(sanos.id) 1]);
histogram(sanos.age)
hold on
histogram(edadSanoEstimado)

title("Edad de las personas sanas")
legend ("Datos reales", "Modelo estimado")
xlabel("Edad (días)")
ylabel("Cantidad de sanos")

clear edadSanoEstimado

%% Gráfico de la altura de las personas sanas versus su estimación
figure
alturaSanoEstimado = normrnd(likeSano(2, 1), likeSano(2,2), [length(sanos.id) 1]);
histogram(sanos.height)
hold on
histogram(alturaSanoEstimado)

title("Altura de las personas sanas")
legend ("Datos reales", "Modelo estimado")
xlabel("Altura (cm)")
ylabel("Cantidad de enfermos")

clear alturaSanoEstimado

%% Gráfico del peso de las personas sanas versus su estimación
figure
pesoSanoEstimado = normrnd(likeSano(3, 1), likeSano(3,2), [length(sanos.id) 1]);
histogram(sanos.weight)
hold on
histogram(pesoSanoEstimado)

title("Peso de las personas sanas")
legend ("Datos reales", "Modelo estimado")
xlabel("Peso (kg)")
ylabel("Cantidad de sanos")

clear pesoSanoEstimado

%
%% 1.4 Evidencia de cada característica

pEdad = [mean(traindata.age) std(traindata.age)]; %./length(traindata.age);
pAltura = [mean(traindata.height) std(traindata.height)]; %./length(traindata.height);
pPeso = [mean(traindata.weight) std(traindata.weight)]; %./length(traindata.weight);

pdfEdad = pdf('Normal', testdata.age, pEdad(1), pEdad(2)); % p(x) / x = Edad
pdfAltura = pdf('Normal', testdata.height, pAltura(1), pAltura(2));  % p(x) / x = altura
pdfPeso = pdf('Normal', testdata.weight, pPeso(1), pPeso(2));  % p(x) / x = Peso

meanEdadSano = mean(sanos.age);
meanAlturaSano = mean(sanos.height);
meanPesoSano = mean(sanos.weight);

stdEdadSano = std(sanos.age);
stdAlturaSano = std(sanos.height);
stdPesoSano = std(sanos.weight);


pdfEdadSano = pdf('Normal', testdata.age, meanEdadSano, stdEdadSano); % p(x) / x = Edad
pdfAlturaSano = pdf('Normal', testdata.height, meanAlturaSano, stdAlturaSano);  % p(x) / x = altura
pdfPesoSano = pdf('Normal', testdata.weight, meanPesoSano, stdPesoSano);  % p(x) / x = Peso

% Variables para la matriz de confusión
tp = 0; % True positives, donde "positive" representa que sufre de una enfermedad
tn = 0; % True negatives
fp = 0; % False positives
fn = 0; % False negatives

%%

predictionEdad = nan(length(testdata.id), 1);

predictionAltura = nan(length(testdata.id), 1);

predictionPeso = nan(length(testdata.id), 1);

testdata = [testdata table(predictionEdad) table(predictionAltura) table(predictionPeso)];

clear predictionEdad predictionAltura predictionPeso

cont = 0; 

%% Aplicamos Bayes
tic 
gEdadSano = prior(1).*pdfEdadSano./pdfEdad;

gAlturaSano = prior(1).*pdfAlturaSano./pdfAltura;

gPesoSano = prior(1).*pdfPesoSano./pdfPeso;

testdata.predictionEdad(gEdadSano>=0.5) = 0;
testdata.predictionEdad(gEdadSano<0.5) = 1;

testdata.predictionAltura(gAlturaSano>=0.5) = 0;
testdata.predictionAltura(gAlturaSano<0.5) = 1;

testdata.predictionPeso(gPesoSano>=0.5) = 0;
testdata.predictionPeso(gPesoSano<0.5) = 1;
toc
%%
tic
% Variables para la matriz de confusión
tp = 0; % True positives, donde "positive" representa que sufre de una enfermedad
tn = 0; % True negatives
fp = 0; % False positives
fn = 0; % False negatives

cont = 0;
    for j = 1:length(testdata.id); % Repeticiones hasta la cantidad de personas en el test 
        % Funciones discriminantes
        if testdata.predictionEdad(j) == testdata.cardio(j)
            cont = cont + 1;
            if testdata.predictionEdad(j) == 0
                tp(1) = tp(1) + 1;
            else
                tn(1) = tn(1) + 1;
            end
        else
            if testdata.predictionEdad(j) == 0
                fp(1) = fp(1) + 1;
            else
                fn(1) = fn(1) + 1;
            end
            
        end   
    end 

aciertosEdad = (tp+tn)/length(testdata.id)


% Variables para la matriz de confusión
tp = 0; % True positives, donde "positive" representa que sufre de una enfermedad
tn = 0; % True negatives
fp = 0; % False positives
fn = 0; % False negatives

    cont = 0;
    for j = 1:length(testdata.id); % Repeticiones hasta la cantidad de personas en el test 
        % Funciones discriminantes
        if testdata.predictionAltura(j) == testdata.cardio(j)
            cont = cont + 1;
            if testdata.predictionAltura(j) == 0
                tp(1) = tp(1) + 1;
            else
                tn(1) = tn(1) + 1;
            end
        else
            if testdata.predictionAltura(j) == 0
                fp(1) = fp(1) + 1;
            else
                fn(1) = fn(1) + 1;
            end
            
        end   
    end 

aciertosAltura = (tp+tn)/length(testdata.id)


% Variables para la matriz de confusión
tp = 0; % True positives, donde "positive" representa que sufre de una enfermedad
tn = 0; % True negatives
fp = 0; % False positives
fn = 0; % False negatives

    cont = 0;
    for j = 1:length(testdata.id); % Repeticiones hasta la cantidad de personas en el test 
        % Funciones discriminantes
        if testdata.predictionPeso(j) == testdata.cardio(j)
            cont = cont + 1;
            if testdata.predictionPeso(j) == 0
                tp(1) = tp(1) + 1;
            else
                tn(1) = tn(1) + 1;
            end
        else
            if testdata.predictionPeso(j) == 0
                fp(1) = fp(1) + 1;
            else
                fn(1) = fn(1) + 1;
            end
            
        end   
    end 

aciertosPeso = (tp+tn)/length(testdata.id)

toc
%% Matrices de confusión
hold off
figure
cmAge = confusionchart(testdata.cardio,testdata.predictionEdad);
title('Matriz de confusión para predicción por edad')
figure
cmHeight = confusionchart(testdata.cardio,testdata.predictionAltura);
title('Matriz de confusión para predicción por altura')
figure
cmWeight = confusionchart(testdata.cardio,testdata.predictionPeso);
title('Matriz de confusión para predicción por peso')

%%





%% Uso de peso y altura como features
tic

pesoAlturaSano = [sanos.weight(:) sanos.height(:)]; % Matriz con peso y altura, no enfermos
pesoAlturaEnfermo = [enfermos.weight(:) enfermos.height(:)]; % Matriz con peso y altura, enfermos

covSano = cov(pesoAlturaSano);
covEnfermo = cov(pesoAlturaEnfermo);

meanSano = mean(pesoAlturaSano)';
meanEnfermo = mean(pesoAlturaEnfermo)';

PesAltTest = [testdata.weight(:) testdata.height(:)];

%%
tic
 % Define the grid for visualization
  [X,Y] = meshgrid(35:120,130:200);
  % Define the constant
  const = (1/sqrt(2*pi))^2;
  const = const/sqrt(det(covSano));
% Compute Density at every point on the grid
  temp = [X(:)-meanSano(1) Y(:)-meanSano(2)];
  pdf = const*exp(-0.5*diag(temp*inv(covSano)*temp'));
  pdf = reshape(pdf,size(X));
% plot the result 
    figure(1)
    RGB1 = 0.2*rand(71,86,3) + 0.6;
    RGB1(:,:,3) = RGB1(:,:,3) + 0.2;
    surfc(X, Y, pdf,'CData', RGB1);

hold on 

 % Define the grid for visualization
  [X,Y] = meshgrid(35:120,130:200);
  % Define the constant
  const = (1/sqrt(2*pi))^2;
  const = const/sqrt(det(covEnfermo));
% Compute Density at every point on the grid
  temp = [X(:)-meanEnfermo(1) Y(:)-meanEnfermo(2)];
  pdf = const*exp(-0.5*diag(temp*inv(covEnfermo)*temp'));
  pdf = reshape(pdf,size(X));
% plot the result 
    figure(1)
    RGB2 = 0.2*rand(71,86,3) + 0.6;
    RGB2(:,:,1) = RGB2(:,:,1) + 0.2;
    surfc(X, Y, pdf,'CData', RGB2);

title("Distribución normal altura-peso")
ylabel("Altura (cm)")
xlabel("Peso (kg)")
zlabel("Cantidad")
legend('Personas sanas', '', 'Personas enfermas', '')
toc
%%
predictionMultVar = nan(length(testdata.id), 1);
predictionNaiveBayes = nan(length(testdata.id), 1);
predictionDep = nan(length(testdata.id), 1);

testdata = [testdata table(predictionNaiveBayes) table(predictionMultVar) table(predictionDep)];

%%
tic
WSano = -0.5*inv(covSano);
WEnfermo = -0.5*inv(covEnfermo);

wSano = inv(covSano)*meanSano;
wEnfermo = inv(covEnfermo)*meanEnfermo;

w0Sano = -0.5*meanSano'*inv(covSano)*meanSano - 0.5*log(det(covSano)) + log(prior(1));
w0Enfermo = -0.5*meanEnfermo'*inv(covEnfermo)*meanEnfermo-0.5*log(det(covEnfermo))+log(prior(2));

cont = 0;
for i = 1:length(testdata.id)
    sano = PesAltTest(i,:)*WSano*PesAltTest(i,:)' + wSano'*PesAltTest(i,:)'+w0Sano;
    enfermo = PesAltTest(i,:)*WEnfermo*PesAltTest(i,:)' + wEnfermo'*PesAltTest(i,:)'+w0Enfermo;
    
    if sano > enfermo
        estado = 0; % estado de la persona es no enferma
    else
        estado = 1; % estado de la persona es enferma
    end
    
    testdata.predictionMultVar(i) = estado;

    % Verificación si la persona está sana o no
    
    if estado == testdata.cardio(i)
        cont = cont + 1;
    end

end

aciertosMultVar = cont/length(testdata.id)

toc
%% Naive Bayes asumiendo que las variables son independientes
tic
carSano = [sanos.age(:) sanos.height(:) sanos.weight(:)]; % Matriz con peso y altura, no enfermos
carEnfermo = [enfermos.age(:) enfermos.height(:) enfermos.weight(:)]; % Matriz con peso y altura, enfermos

meanSano = mean(carSano);
meanEnfermo = mean(carEnfermo);

DesEstSano = std(carSano);
DesEstEnfermo = std(carEnfermo);

cont = 0;
for i = 1:length(testdata.id)
    X = [testdata.age(i) testdata.height(i) testdata.weight(i)];
    sano = -0.5*sum((((X-meanSano)./DesEstSano)').^2)+log(prior(1));
    enfermo = -0.5*sum((((X-meanEnfermo)./DesEstEnfermo)').^2)+log(prior(2));
    
    if sano > enfermo
        estado = 0;% Persona no está enferma
    else
        estado = 1;% Persona está enferma
    end

    testdata.predictionNaiveBayes(i) = estado;
    
    if estado == testdata.cardio(i)
        cont = cont + 1;
    end
end

aciertosNaiveIndep = cont/length(testdata.id)

toc
%% Naive Bayes asumiendo que las variables son dependientes
tic
meanSano = mean(carSano)';
meanEnfermo = mean(carEnfermo)';

covSano = cov(carSano);
covEnfermo = cov(carEnfermo);

RSano = corrcov(covSano);
REnfermo = corrcov(covEnfermo);

mean3 = mean([traindata.age traindata.height traindata.weight])';
cov3 = cov([traindata.age traindata.height traindata.weight]);


NaiveTest = [testdata.age(:) testdata.height(:) testdata.weight(:)];

pdf3Enfermo = mvnpdf(NaiveTest, meanEnfermo', covEnfermo);
pdf3 = mvnpdf(NaiveTest, mean3', cov3);

g3Enfermo = prior(2).*pdf3Enfermo./pdf3;

testdata.predictionDep(g3Enfermo>=0.5) = 1;
testdata.predictionDep(g3Enfermo<0.5) = 0;
toc


% Variables para la matriz de confusión
tp = 0; % True positives, donde "positive" representa que sufre de una enfermedad
tn = 0; % True negatives
fp = 0; % False positives
fn = 0; % False negatives

cont = 0;
    for j = 1:length(testdata.id); % Repeticiones hasta la cantidad de personas en el test 
        % Funciones discriminantes
        if testdata.predictionDep(j) == testdata.cardio(j)
            cont = cont + 1;
            if testdata.predictionEdad(j) == 0
                tp(1) = tp(1) + 1;
            else
                tn(1) = tn(1) + 1;
            end
        else
            if testdata.predictionEdad(j) == 0
                fp(1) = fp(1) + 1;
            else
                fn(1) = fn(1) + 1;
            end
            
        end   
    end 

aciertosDep = (tp+tn)/length(testdata.id)

toc
%%
tic
WSano = -0.5*pinv(covSano);
WEnfermo = -0.5*pinv(covEnfermo);

wSano = pinv(covSano)*meanSano;
wEnfermo = pinv(covEnfermo)*meanEnfermo;

w0Sano = -0.5*meanSano'*pinv(covSano)*meanSano - 0.5*log(abs(det(covSano))) + log(prior(1));
w0Enfermo = -0.5*meanEnfermo'*pinv(covEnfermo)*meanEnfermo-0.5*log(abs(det(covEnfermo))) + log(prior(2));



cont = 0;
for i = 1:length(testdata.id)
    sano = NaiveTest(i,:)*WSano*NaiveTest(i,:)' + wSano'*NaiveTest(i,:)' + w0Sano;
    enfermo = NaiveTest(i,:)*WEnfermo*NaiveTest(i,:)' + wEnfermo'*NaiveTest(i,:)' + w0Enfermo;


    sano;
    enfermo;

    aiuda(i, [1 2]) = [sano enfermo];

    if sano > enfermo
        estado = 0;% Persona no está enferma
    else
        estado = 1;% Persona está enferma
    end
    
    if estado == testdata.cardio(i)
        cont = cont + 1;
    end
end
aciertosDep2 = cont/length(testdata.id)

toc