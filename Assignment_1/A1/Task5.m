%%%% This code is written for Task 5 part


%% First load the data
Data = load('A1_data.mat')
t_vector = getfield(Data,'t'); %% sum of two sin function (freq of 1/20 and 1/5) plus gaussian noise (50 X 1)
X_regression = getfield(Data,'X'); %% (50 X 1000) [ui vi]  500 pairs of 
X_inter      = getfield(Data,'Xinterp');
time_inter   = getfield(Data,'ninterp');
%% Initialization
lambdavec = exp( linspace( log(0.01), log(10), 10))
what = [];
Reconstructed_y = [];
Interpolated_y = [];
time = [0:49];
%%

%% initialization for k-fold validation
K = 10;



[wopt,lambdaopt,RMSEval,RMSEest] = skeleton_lasso_cv(t_vector,X_regression,lambdavec,K);
woptzero = sum(wopt~=0)
%% plot task5_b

figure;
plot(lambdavec, RMSEval,'r')
hold on 
plot(lambdavec, RMSEest,'b')
hold on 
plot(lambdaopt*ones(1,13),[0:0.5:6],'g')
legend('$RMSE_{val}$','$RMSE_{est}$','$\hat{\lambda}$')



lambda = [0.1 lambdaopt 10]
%% Running ccd and Reconstructed data points
for i = 1:3
    wold = zeros(1000,1);
    what= [what,skeleton_lasso_ccd(t_vector,X_regression,lambda(i),wold)];
    Reconstructed_y = [Reconstructed_y, (X_regression*what(:,i))];
    Interpolated_y  = [Interpolated_y, X_inter * what(:,i)];
end

nonzero = sum(what(:,2)~=0)

%%
%% plot part task5_b
%close all;

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







