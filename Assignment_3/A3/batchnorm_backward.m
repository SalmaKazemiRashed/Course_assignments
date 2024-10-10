function [dx, dgamma, dbeta] = batchnorm_backward( X,dy, gamma,beta)

eps = 1e-3;
sz = size(X);
batch = sz(end);
features = prod(sz(1:end-1));

X_reshape = reshape(X,[features, batch]);
mu = 1./features*sum(X_reshape,1);
var = 1./features*sum((X_reshape-mu).^2, 1);

dy_reshape =reshape(dy,[features, batch]);
dbeta = sum(dy_reshape,1);
dgamma = sum((X_reshape-mu) .* (var + eps).^(-1. / 2.) .* dy_reshape, 1);
dh = (1/ features) .* gamma .* (var + eps).^(-1./ 2) .* (features * dy_reshape -sum(dy_reshape,1) -...
        (X_reshape - mu) .* (var + eps).^(-1.0) .* sum(dy_reshape .* (X_reshape - mu), 1));
   
dx = reshape(dy,[sz(1:end-1),batch] );

end



  