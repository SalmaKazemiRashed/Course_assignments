Data = load('A2_data.mat')
train_data = getfield(Data, 'train_data_01');
train_label = getfield(Data, 'train_labels_01');
test_data = getfield(Data, 'test_data_01');
test_label = getfield(Data, 'test_labels_01');
size(train_data)
size(train_label)

size(test_data)
size(test_label)


num_of_zero_tr = sum(train_label==0)
num_of_one_tr = sum(train_label==1)
num_of_zero_test = sum(test_label==0)
num_of_zero_tst = sum(test_label==1)



%% Normalize_data
normalized_train_data = (train_data-mean(train_data));%./(sqrt(var(train_data)));
normalized_test_data = (test_data-mean(test_data));%./(sqrt(var(test_data)));

%%%
svmmodel = fitcsvm(normalized_train_data',train_label);
labels   =  predict(svmmodel,normalized_train_data');

%labels(labels==2)=0;
%labels(labels==1)=1;

num_class_tr_zero = sum(labels==0)
num_class_tr = sum(labels==1)

n_of_0_in_0 = sum([labels(labels==0)==train_label(labels==0)]);
n_of_1_in_0 = sum(labels==0)-n_of_0_in_0;
n_of_1_in_1 = sum(labels(labels==1)==train_label(labels==1)) ;
n_of_0_in_1 = sum(labels==1)-n_of_1_in_1;





n_of_0_misclassification = sum(labels(labels==0)~=train_label(labels==0)) ;
n_of_1_missclassification= sum(labels(labels==1)~=train_label(labels==1));

%%
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
title('True lables')
legend('\bf{1}','\bf{0}')
%%%%%%



g_1 = dataInPrincipalComponentSpace (labels==1,1:2);
g_2  =dataInPrincipalComponentSpace (labels==0,1:2);


figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')
title 'Linear SVM classifier labels'
legend('\bf{1}','\bf{0}')



%% Test data

t_labels   =  predict(svmmodel,normalized_test_data');


%t_labels(t_labels==2)=0;
%t_labels(t_labels==1)=1;


num_class_test_zero = sum(t_labels==0)
num_class_test = sum(t_labels==1)


test_n_of_0_in_0 = sum([t_labels(t_labels==0)==test_label(t_labels==0)]);
test_n_of_1_in_0 = sum(t_labels==0)-test_n_of_0_in_0;
test_n_of_1_in_1 = sum(t_labels(t_labels==1)==test_label(t_labels==1)) ;
test_n_of_0_in_1 = sum(t_labels==1)-test_n_of_1_in_1;



%% test plot
[U,S,V] = svd(normalized_test_data);%Singular value decomposition

d_2 = U(:,1:2);

dataInPrincipalComponentSpace        = zeros(2115,2);
dataInPrincipalComponentSpace(:,1) = normalized_test_data'*d_2(:,1);
dataInPrincipalComponentSpace(:,2) = normalized_test_data'*d_2(:,2);

g_1 = dataInPrincipalComponentSpace (t_labels==1,1:2);
g_2  =dataInPrincipalComponentSpace (t_labels==0,1:2);


figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')
title 'Linear SVM classifier labels'
legend('\bf{1}','\bf{0}')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[U,S,V] = svd(normalized_test_data);%Singular value decomposition

d_2 = U(:,1:2);

dataInPrincipalComponentSpace        = zeros(2115,2);
dataInPrincipalComponentSpace(:,1) = normalized_test_data'*d_2(:,1);
dataInPrincipalComponentSpace(:,2) = normalized_test_data'*d_2(:,2);

g_1 = dataInPrincipalComponentSpace (test_label==1,1:2);
g_2  =dataInPrincipalComponentSpace (test_label==0,1:2);


figure;
scatter(g_1(:,1),g_1(:,2),'b*')
hold on
scatter(g_2(:,1),g_2(:,2),'ro','filled')
title 'True labels'
legend('\bf{1}','\bf{0}')








