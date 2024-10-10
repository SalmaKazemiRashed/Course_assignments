%% Task 1
x_vector = [-2 -1 1 2];
phi_x      = [x_vector; x_vector.^2]
K = zeros(4,4);

for i =1:4
    for j = 1:4
        K(i,j) = phi_x(:,i)'*phi_x(:,j);
    end
end
%%



y_vector = [1 -1 -1 1];
y_iy_j = zeros(4,4);
%% Task2
for i =1:4
    for j = 1:4
        y_iy_j(i,j) = y_vector(i)*y_vector(j);
    end
end
  
alpha = 4./sum(sum(y_iy_j.*K))
%%

%% Exhaustive search
a = 0:0.001:0.4;
y = 4.*a-a.*a/2* (sum(sum(y_iy_j.*K)))
plot(a,y )
[value, index] = max(y)
a(index)
%%






%% Task3: b calculation
sum_xs = 0;
for i =1:4
    sum_n = 0;
    for j =1:4
        sum_n = sum_n + a(index) * y_vector(j)*K(j,i);
    end
    sum_xs = sum_xs +y_vector(i)-sum_n;
end
b = sum_xs/4
%%

%%
figure;
x = -4:0.01:4;
plot(x,0.666*x.^2-1.6650) 
hold on
scatter ([-1,1],[0,0],'filled','r')
hold on
scatter ([-2,2],[0,0],'filled','g')

%%


%% Task4
x_vector = [-3 -2 -1 0 1 2 4];
phi_x      = [x_vector; x_vector.^2]
K = zeros(7,7);

for i =1:7
    for j = 1:7
        K(i,j) = phi_x(:,i)'*phi_x(:,j);
    end
end
K
a(index)
y_vector = [+1 +1 -1 -1 -1 +1 +1]
a = [0 0.1111 0.1111 0 0.1111 0.1111 0]
%%
sum_xs = 0;
for i =[2,3,5,6]
    sum_n = 0;
    for j =1:7
        sum_n = sum_n +  a(j)* y_vector(j)*K(j,i);
    end
    sum_xs = sum_xs +y_vector(i) -sum_n;
end
b = sum_xs/4

figure;
x = -4:0.01:4;
plot(x,0.666*x.^2+b) 
hold on
scatter ([-1,0,1],[0,0,0],'filled','r')
hold on
scatter ([-3,-2,2,4],[0,0,0,0],'filled','g')


