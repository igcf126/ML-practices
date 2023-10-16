%% Prerrequisitos
% 1092228, Ian Grabriel Cañas Fernández

%% Sección I - Introducción a MATLAB
%%
% *Programming and scripts* 
%
% Grafica una esfera de radio r.
[x,y,z] = sphere;       % Esfera unitaria.
r = 2;
surf(x*r,y*r,z*r)       % ajusta cada dimensión y grafica.
axis equal              % Usa una escala igual para los ejes. 
 
% Encuentra el área y volumen.
A = 4*pi*r^2;
V = (4/3)*pi*r^3;

%%

N = 100;
f(1) = 1;
f(2) = 1;

for n = 3:N
    f(n) = f(n-1) + f(n-2);
end
f(1:10)

num = randi(100)
if num < 34
   sz = 'low'
elseif num < 67
   sz = 'medium'
else
   sz = 'high'
end

%% 
% *Matrices and arrays*
%
a = [1 2 3 4]
a = [1 3 5; 2 4 6; 7 8 10]
z = zeros(5,1)

%%

a + 10
sin(a)
a'
p = a*inv(a)

format long
p = a*inv(a)

format short
p = a.*a

a.^3

%%

A = [a,a]
A = [a; a]

%%

sqrt(-1)
c = [3+4i, 4+3j; -i, 10j]

%%
% *Array indexing*
%
A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
A(4,2)
A(8)
A(4,5) = 17
A(1:3,2)
A(3,:)
B = 0:10:100

%%
% *Calling functions*
%
A = [1 3 5];
max(A)

B = [3 6 9];
union(A,B)

maxA = max(A)

[minA,maxA] = bounds(A)

disp("hello world")

%%
% *2-D and 3-D plots*
%
x = linspace(0,2*pi);
y = sin(x);
plot(x,y)

xlabel("x")
ylabel("sin(x)")
title("Plot of the Sine Function")

plot(x,y,"r--")

x = linspace(0,2*pi);
y = sin(x);
plot(x,y)

hold on

y2 = cos(x);
plot(x,y2,":")
legend("sin","cos")

hold off

%%

x = linspace(-2,2,20);
y = x';
z = x .* exp(-x.^2 - y.^2);

surf(x,y,z)

%%

t = tiledlayout(2,2);
title(t,"Trigonometric Functions")
x = linspace(0,30);

nexttile
plot(x,sin(x))
title("Sine")

nexttile
plot(x,cos(x))
title("Cosine")

nexttile
plot(x,tan(x))
title("Tangent")

nexttile
plot(x,sec(x))
title("Secant")


%% I. MANIPULANDO MATRICES.
clear 
clc
%%
% *1) Realice las transformaciones lineales $$ (a) q=Qx $$ y $$ (b)
% x=Q^{-1}q $$, donde*
% *Comente sus observaciones.* 

format rational
Q = [5/6 1/6 0; 5/6 0 1/6; 0 5/6 1/6]
format default
x = [56; 14; 0.005]

q = Q*x
x = inv(Q)*q

%%
% Claramente apreciamos el cumplimiento de los conceptos de álgebra lineal
% en cuando al producto y la matriz inversa.

%%
% *2) Importe los datos de un archivo .csv, .xls o .xlsx como una matrix a 
% su plataforma de preferencia y despliegue las primeras 15 filas y 4 
% columnas.* 

A = readmatrix('Matriz.xlsx');
A(1:15,1:4)

%%
% *3) Cree un nuevo vector columna $$ x=[x^1 x^2 \dots x^{20}] $$ y copie 
% allí el contenido de los últimos 20 elementos de la tercera columna de 
% la matriz creada en el paso anterior.* 

X = A(end-19:end,3)

%%
% *4) Cree un nuevo vector fila $$ y=[y^1 y^2 \dots y^5 ]^T $$ , donde 
% cada $$ y^i \sim \mathcal{N} (1.7,5)   $$ es un número aleatorio generado según la 
% distribución Gaussiana especificada.* 

Y = transpose(normrnd(1.7, 5, 5, 1)) % normrnd(mu,sigma,sz) genera un array de números rándom de distribución normal, donde el vector sz especifica el tamaño de la matriz Y.


%%
% *5) Genere la matriz A = xy.

A = X*Y

%%
% *6) Aplique un comando de reshape para convertir la matriz A en un s´olo 
% vector columna a* 

a = reshape(A, [], 1)

%% II. VECTORIZANDO OPERACIONES.
%%
% *1) Compute la suma de los elementos de cada columna de la matriz A.* 

S = sum(A)

%%
% *2) Compute el promedio de los elementos de cada columna de la matriz A.* 

P = median(A)

%%
% *3) Compute la varianza de los elementos de cada columna de la matriz A.* 

va = var(A)

%%
% *4) Compute el vector media a $$a$$ partir de los vectores fila de 
% la matriz A.*

a = median(transpose(A))

%%
% *5) Compute la distancia euclideana entre $$a$$ y cada vector fila de la 
% matriz A.* 

d = pdist2(transpose(A),a,"euclidean")

%%
% *6) Vectorice e implemente las siguientes combinaciones lineales. 
% Asuma los valores de su preferencia para los coeficientes $$ a^i $$, 
% $$ a^{ji} $$ y los descriptores x^i.* 

za = a(1)
for i = 2:5
    za = za + X(i)*a(i);
end

za

%%

for j=1:3
    zb(j) = A(j);
    for i = 2:5
        zb(j) = zb(j) + X(i)*A(j,i);
    end
end 
zb

%% III. OTRAS OPERACIONES ÚTILES.
%%
% *1) Aplique una función de slicing para extraer una matriz C de tamaño 
% 4x4 a partir de las primeras 4 filas y 4 columnas de la matriz A.* 

C = A(1:4,1:4)

%%
% *2) Compute la matriz de _eigenvalues_ $$ \lambda $$ y la matriz de 
% _eigenvectors_ $$ V $$ de la matriz C. Regenere a $$ C=V \lambda V^{-1} $$* 

[V, lambda] = eig(C) % V = eigenvectors; lambda = eigenvalues

C = V*lambda*inv(V)

%%
% *3) Lleve a cabo la operación de _singular value decomposition_ sobre las 
% matrices A y C. Regenere las matrices A y C utilizando las matrices 
% resultantes de la descomposición.* 

[UA,SA,VA] = svd(A)
A = UA*SA*VA


%%
[UC,SC,VC] = svd(C)
C = UC*SC*VC