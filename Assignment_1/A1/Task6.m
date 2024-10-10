%% First load the data
Data = load('A1_data.mat')
Ttest = getfield(Data,'Ttest'); %% size = 24697x1
Ttrain = getfield(Data,'Ttrain'); %%19404x 1
fs = getfield(Data,'fs');
Xaudio = getfield(Data,'Xaudio'); %%352*2000

soundsc(Ttrain,fs)
%soundsc(Ttest,fs)


%% 
lambdavec = exp( linspace( log(.0001), log(1), 10))

%%

%% initialization for k-fold validation
K = 3;
z = randperm(2000);
X_audio_sim = Xaudio(:,z(1:500));
size(X_audio_sim)
[Wopt,lambdaopt,RMSEval,RMSEest] = skeleton_multiframe_lasso_cv(Ttrain,X_audio_sim,lambdavec,K);


figure;
plot(lambdavec, RMSEval,'r')
hold on 
plot(lambdavec, RMSEest,'b')
hold on 
plot(lambdaopt*ones(1,7),[0:0.01:0.06],'g')
legend('$RMSE_{val}$','$RMSE_{est}$','$\hat{\lambda}$')
