%% Asignación 1
% 
clc, clear
hold off
tic

n = 300;
k=5;

for i = 1:10
    mu(i) = rand*100;
    desvEst(i) = rand*15;
    varianza(i) = desvEst(i)^2;
    X(:,i) = normrnd(mu(i), desvEst(i), n, 1);
end

Datos = [X(:,1) X(:,2); X(:,3) X(:,4); X(:,5) X(:,6); X(:,7) X(:,8); X(:,9) X(:,10)];

plot(X(:,1),X(:,2), '.k',X(:,3),X(:,4), '.k',X(:,5),X(:,6), '.k',X(:,7),X(:,8), '.k',X(:,9),X(:,10), '.k')

%%
tic
figure
m = rand(k,2)*100;


hold on
plot(m(:, 1), m(:, 2), '^b')

dentro = true;
iteraciones = 0;
while dentro == true
% for z=1:100   
    iteraciones = iteraciones +1;
    r = m;
    B = zeros(n, k);
    for i = 1:length(Datos(:,1))

            d = sqrt(sum(transpose(Datos(i, :) - m).^2));

        [dis, pos] = mink(d, 1);
        
        B(i, pos) = 1;
    end

    for l = 1:k
        m(l, :) = sum(B(:,l).*Datos)/sum(B(:,l));
    end
    %z
    %plot(m(:, 1), m(:, 2), 'o')\
%     e = sqrt(sum((nansum(abs(m-r))/k).^2));
    e = nansum(abs(m-r))/k;
    if e < 0.001 
        dentro = false;
        iteraciones
    end
end

b2 = reshape(B, [], 1);
C = nan(n*k, k*2);

for a = 1:k*2
% A = 1:10
% [round(A/2); ~(mod(A, 2))+1]
    
    newB = (Datos(logical(reshape(B(:,round(a/2)), [],1)), ~(mod(a, 2))+1)); % round(k/2) representa cada valor de k, y a representa x y y de cada valor de k, por eso el round(k/2)
    newA = [newB; transpose(nan(1, n*k-length(newB)))];
    C(:, a) = newA;

end

% plot(m(:, 1), m(:, 2), 'o')
hold off

plot(C(:,1), C(:,2), '.m',C(:,3),C(:,4), '.b',C(:,5),C(:,6), '.r',C(:,7),C(:,8), '.g',C(:,9),C(:,10), '.c');
hold on
%plot(C(:,1), C(:,2), '.k')
plot(m(:, 1), m(:, 2), '.k')

% fprintf ('Centroides en:\n')
x = m(:,1);
y = m(:,2);

Centroides = table(x, y)

toc 