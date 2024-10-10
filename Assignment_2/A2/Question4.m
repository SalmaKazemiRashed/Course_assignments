
Data = load('A2_data.mat')
train_data = getfield(Data, 'train_data_01');
train_label = getfield(Data, 'train_labels_01');
size(train_data)
size(train_label)

%% Normalize_data
normalized_train_data = (train_data-mean(train_data));%./(sqrt(var(train_data)));
%hist(train_data(:,1))
%hist(normalized_train_data(:,1))
mean(normalized_train_data(:,2));
var(normalized_train_data(:,2));
 
K=2;

[y,C] = K_means_clustering(normalized_train_data,K);

% 
% 
[U,S,V] = svd(normalized_train_data);%Singular value decomposition

d_2 = U(:,1:2);

dataInPrincipalComponentSpace        = zeros(12665,2);
dataInPrincipalComponentSpace(:,1) = normalized_train_data'*d_2(:,1);
dataInPrincipalComponentSpace(:,2) = normalized_train_data'*d_2(:,2);

g_1 = dataInPrincipalComponentSpace (y==1,1:2);
g_2  =dataInPrincipalComponentSpace (y==2,1:2);
% 
% 
figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')
legend('\bf{1}','\bf{2}')


figure;title 'K=2'
C_image = zeros(K,28,28);
for k=1:K
    C_image(k,:,:) = reshape(C(:,k),28,28)
    subplot(1,K,k)
    imshow(reshape(C(:,k),28,28))
     xlabel(num2str(k))
end




%% K=5
K=5;

[y,C] = K_means_clustering(normalized_train_data,K);

% 

[U,S,V] = svd(normalized_train_data);%Singular value decomposition

d_2 = U(:,1:2);

dataInPrincipalComponentSpace        = zeros(12665,2);
dataInPrincipalComponentSpace(:,1) = normalized_train_data'*d_2(:,1);
dataInPrincipalComponentSpace(:,2) = normalized_train_data'*d_2(:,2);

g_1 = dataInPrincipalComponentSpace (y==1,1:2);
g_2  =dataInPrincipalComponentSpace (y==2,1:2);
g_3 = dataInPrincipalComponentSpace (y==3,1:2);
g_4  =dataInPrincipalComponentSpace (y==4,1:2);
g_5 = dataInPrincipalComponentSpace (y==5,1:2);

figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')
hold on
scatter(g_3(:,1),g_3(:,2),'gx')
hold on
scatter(g_4(:,1),g_4(:,2),'y^','filled')
hold on
scatter(g_5(:,1),g_5(:,2),'cyan','filled')
legend('\bf{1}','\bf{2}','\bf{3}','\bf{4}','\bf{5}')

figure;
C_image = zeros(K,28,28);
title 'K=5'
for k=1:K
    C_image(k,:,:) = reshape(C(:,k),28,28)
    subplot(1,K,k)
    imshow(reshape(C(:,k),28,28))
   xlabel(num2str(k))
end



