%%%% This code is written for Task 4 part


%% First load the data
Data = load('A1_data.mat')
t_vector = getfield(Data,'t'); %% sum of two sin function (freq of 1/20 and 1/5) plus gaussian noise (50 X 1)
X_regression = getfield(Data,'X'); %% (50 X 1000) [ui vi]  500 pairs of 
X_inter      = getfield(Data,'Xinterp');
time_inter   = getfield(Data,'ninterp');
%% Initialization
lambda = [0.1, 3, 10];
what = [];
Reconstructed_y = [];
Interpolated_y = [];
time = [0:49];
%%

%% Running ccd and Reconstructed data points
for i = 1:3
    wold = zeros(1000,1);
    what= [what,skeleton_lasso_ccd(t_vector,X_regression,lambda(i),wold)];
    Reconstructed_y = [Reconstructed_y, (X_regression*what(:,i))];
    Interpolated_y  = [Interpolated_y, X_inter * what(:,i)];
end
%%
%% plot part task4_a
close all;

for i = 1:3
    subplot(3,1,i)
    plot(time,t_vector,'ro')
    hold on
    scatter (time, Reconstructed_y(:,i),'gx')
    hold on
   
    hold on
    plot(time_inter, Interpolated_y(:,i))
    legend('Original', 'Reconstruted','Interpolated')
end
figure;
for i = 1:3
    subplot(3,1,i)
    plot(what(:,i))
    nn = what(:,i)
    hold on
    nn = sum(what(:,i)~=0)
    title(['Number of Non-zero w =',num2str(nn)])
end

%% actual number of nonzero coordinates task4_b
n = [0:49]';
fi = [0.02, 0.05, 0.1, 0.2] %% we only need 1/20 and 1/5 frequencies

X_reg_new = []
for k=1:length(fi)
     x_temp = [sin(2*pi*fi(k).*n) cos(2*pi*fi(k).*n)];
     X_reg_new = [X_reg_new, x_temp];
end

actual_y =  5*cos(2*pi*(n/20+1/3))+2*cos(2*pi*(n/5-1/4));
w_LS     = (X_reg_new'*X_reg_new)^(-1)*X_reg_new'*actual_y


%% plot 
figure; 
stem(w_LS)
nn = sum(double(w_LS)~=0 )
title(['Number of Non-zero w =',num2str(nn)])

figure;
plot(n,actual_y,'bo')
hold on
plot(n, X_reg_new*w_LS,'r')
