clear, clc

train_set = readtable("Train2.xlsx");
test_set = readtable("Test2.xlsx");

g1 = train_set(train_set.gender==1, :);
g2 = train_set(train_set.gender==2, :);

plot3(g1.height, g1.weight, g1.alco, '.b')
hold on 
plot3(g2.height, g2.weight, g2.alco, '.y')

k = input('Introdutrazca la cantidad de vecinos: ');

nTrainData = length(train_set.height) %size(train_set);
ValReales = test_set.gender;

for i = 1:length(test_set.height)
    %i
    
    Rep = repmat([test_set.height(i), test_set.weight(i), test_set.alco(i)], nTrainData, 1);
    d = ((Rep - [train_set.height(:) train_set.weight(:) train_set.alco(:)]).^2);
    d = sqrt(d(:,1)+d(:,2));
    
    [dis pos] = sort(d,'ascend');
    kNN=pos(1:k);
    kND=dis(1:k);

    c1 = 0;
    c2 = 0;
    for m = 1:k
        if (train_set.gender(pos(m))==1)
            c1 = c1+1;
        elseif (train_set.gender(pos(m))==2)
            c2 = c2+1;
        end
    end

    if c1>c2
        test_set.gender(i)=1;
    elseif c2>c1
        test_set.gender(i)=2;
    end

end

p1 = test_set(test_set.gender==1, :);
p2 = test_set(test_set.gender==2, :);

plot3(p1.height, p1.weight, p1.alco, 'c^')
plot3(p2.height, p2.weight, p2.alco, 'rs')

legend('Género 1','Género 2','Prediction as gender 1','Prediction as gender 2')
title('Resultados kNN')
xlabel('Altura')
ylabel('Peso')
zlabel('Alcohol')

%Accuracy/Precisión
acc = sum(ValReales==test_set.gender)/length(test_set.gender)



