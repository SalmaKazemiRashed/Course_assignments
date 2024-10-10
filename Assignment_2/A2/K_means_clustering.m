function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data D   X = 784x12665
%           K   Number of clusters                    K=2, 5
%
% Output:   y   Nx1 vector of cluster assignments    N=12665
%           C   DxK matrix of cluster centroids       C = 784xK

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax
    tic
    % Step 1: Assign to clusters
        y = step_assign_cluster(X,C);
    % Step 2: Assign new clusters
       C = step_compute_mean(X,y,K);
    if fcdist(C,Cold) < conv_tol
        return
    end
    Cold = C;
    toc
    kiter
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

%% fcdist
function d = fcdist(C1,C2)
    d = sqrt((C1-C2)'*(C1-C2));
end
%%

%% step_assign_cluster
function y = step_assign_cluster(X,C)
    [D,K] = size(C);
    [D,N] = size(X);
    y = [];
    for l = 1:N
        d = fxdist(X(:,l),C);
        [val,y_temp] = min(d);
        y = [y, y_temp];
    end
end

%%

%% step_compute_mean
function C = step_compute_mean(X,y,K)
    [D,N] = size(X);
    C = [] ;
    for k = 1:K
        Nk    = sum(y==k);
        ck     = sum(X(:,y==k),2)/Nk;
        C = [C,ck];
    end
end
%%