clear 
clc

train_set = readtable("Train2.xlsx")
test_set = readtable("Test2.xlsx")

%plot(train_set.height, train_set.weight, '.')

g1 = train_set(train_set.gender==1, :);
g2 = train_set(train_set.gender==2, :);

%i1 = find(g1.gender==train_set.gender)

plot(g1.height, g1.weight, 'r^')
hold on 
plot(g2.height, g2.weight, 'bs')

k = input('Introduzca la cantidad de vecinos: ');

for i = 1:length(test_set.height_sample) % i son las columnas, que representan los datos de cada punto de prueba
    for j = 1:length(train_set.id) % j representa las filas, cada observación conocida
        d(j,i) = pdist([test_set.height_sample(i), test_set.weight_sample(i); train_set.height(j), train_set.weight(j)],'euclidean');
    end
end
    [dn, index] = sort(d);
    result      = d(index(1:k,:)); %d(sort(index(1:k,:)));

for i = 1:length(test_set.height_sample)
    c1 = 0;
    c2 = 0;
    for j = 1:k
        if (train_set.gender(index(j, i))==1)
            c1 = c1+1;
        elseif (train_set.gender(index(j, i))==2)
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

%i1 = find(g1.gender==train_set.gender)

plot(p1.height_sample, p1.weight_sample, 'y^')
hold on 
plot(p2.height_sample, p2.weight_sample, 'ys')