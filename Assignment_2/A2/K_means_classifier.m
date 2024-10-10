function labels = K_means_classifier(X,C)
[D,N] = size(X);
labels = []
for k = 1:N
    [val, ind] = min([fxdist(X(:,k),C(:,1)),fxdist(X(:,k),C(:,2))] );
    labels = [labels,ind];
end
end


%% fxdist
function d = fxdist(x,C)    
    d_temp = [];
    [D,K] = size(C);
    for i =1:K
        d_temp = [d_temp, sqrt((x-C(:,i))'*(x-C(:,i)))];
    end
    d = d_temp;
end
%%