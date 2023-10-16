clc
clear

I = imread('highway.jpg');
I = int32(I);

%size('highway.jpg')
Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);

Irmoda = modefilt(I(:, :, 1), [9,9]);
Igmoda = modefilt(I(:, :, 2), [9,9]);
Ibmoda = modefilt(I(:, :, 3), [9,9]);
%colormap([Ir Ig Ib])

%k = input('Introduzca la cantidad de colores: ');
k=5;
m = rand(k,3)*256;
m = int32(m);
[filas, columnas, capas] = size (I);
IkMeans = int32(nan(filas, columnas, capas));

IModa = int32(nan(filas, columnas, capas));

dentro = true;
iteraciones = 0;

while dentro == true
    iteraciones = iteraciones + 1;
%     dentro = false;
    B = zeros(filas, columnas, k);
    B = int32(B);
    r = m;
    for i = 1:filas
        for j = 1:columnas
            d = sqrt(sum(transpose((reshape(I(i, j, :), 1,3) - m).^2)));
            [dis, pos] = mink(d, 1);
            B(i, j, pos) = 1;
        end
    end
    for l = 1:k
        m(l, :, :) = sum(B(:,:,l).*I, [1 2])/sum(B(:,:,l), [1 2]);
    end
    e = sqrt(sum((nansum(abs(m-r))/k).^2));
    if e < 0.01 
        dentro = false;
        iteraciones
        for i = 1:filas
            for j = 1:columnas
                mi = m(logical(reshape(B(i, j,:),k,1)), :);
                IkMeans(i,j,:) = mi;
            end
        end
    end
end

I = uint8(I);
IkMeans = uint8(IkMeans);
IModa(:,:,1) = Irmoda;
IModa(:,:,2) = Igmoda;
IModa(:,:,3) = Ibmoda;
IModa = uint8(IModa);

figure 
imshow(I)
title('Imagen original')

figure
imshow(IkMeans)
title('kMeans aplicado')

figure
imshow(IModa)
title('Filtro de moda')
