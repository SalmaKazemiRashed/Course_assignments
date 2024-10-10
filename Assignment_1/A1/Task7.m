%% First load the data
Data = load('A1_data.mat')
Ttest = getfield(Data,'Ttest'); %% size = 24697x1
Ttrain = getfield(Data,'Ttrain'); %%19404x 1
fs = getfield(Data,'fs');
Xaudio = getfield(Data,'Xaudio'); %%352*2000


Yclean = lasso_denoise(Ttest,Xaudio,0.003)
 
soundsc(Yclean,fs);