clear
clc

% Se leen los datos del csv y se aleatoriza para prevenir cualquier
% influencia por un orden que se desconozca.

tic
data = readtable("train.csv"); % "Data" porque son los datos que nos dan
traindata = data(randperm(length(data.Name)), :); % "Dataset" porque son los datos con los que trabajaremos

testdata = readtable("test.csv");
tester = readtable("gender_submission.csv");

vivos = traindata(traindata.Survived==1, :);
muertos = traindata(traindata.Survived==0, :);


clear data
toc
%%

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
    kNNAccuracy = sum(testdata.MLSurvived==tester.Survived(:))/length(tester.Survived)
end

toc