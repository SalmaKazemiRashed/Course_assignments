%% This code is for question3


Data = load('A2_data.mat')
train_data = getfield(Data, 'train_data_01');
train_label = getfield(Data, 'train_labels_01');
size(train_data)
size(train_label)

%% Normalize_data
normalized_train_data = (train_data-mean(train_data));%./(sqrt(var(train_data)));
mean(normalized_train_data(:,2))
var(normalized_train_data(:,2))


%% PCA
[coeff,score,latent,tsquared,explained,mu] = pca(normalized_train_data);

%%
%covarianceMatrix = cov(normalized_train_data);
%[V, D] = eig(covarianceMatrix);

PCAs = coeff(:,1:2);

%%

size( normalized_train_data')




group_1 = PCAs(train_label==1,1:2);
group_2  = PCAs(train_label==0,1:2);

figure;
scatter(group_1(:,1),group_1(:,2),'b*')
hold on
scatter(group_2(:,1),group_2(:,2),'ro','filled')
title 'PCA'


%figure;biplot(coeff(:,1:2),'scores',score(:,1:2));


%% SVD

[U,S,V] = svd(normalized_train_data);%Singular value decomposition

d_2 = U(:,1:2);

dataInPrincipalComponentSpace        = zeros(12665,2);
dataInPrincipalComponentSpace(:,1) = normalized_train_data'*d_2(:,1);
dataInPrincipalComponentSpace(:,2) = normalized_train_data'*d_2(:,2);

g_1 = dataInPrincipalComponentSpace (train_label==1,1:2);
g_2  =dataInPrincipalComponentSpace (train_label==0,1:2);


figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')















