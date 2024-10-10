function [Y] = batchnorm_forward(X, gamma, beta)


eps = 1e-3;
sz = size(X);
batch = sz(end);
features = prod(sz(1:end-1));

X_reshape = reshape(X,[features, batch]);
mu = 1./features*sum(X_reshape,1); 
var = 1./features*sum((X_reshape-mu).^2, 1); 



hath = (X_reshape-mu).*(var+eps).^(-1./2);

 out = hath.*gamma+beta ;
Y = reshape(out,[sz(1:end-1),batch]);
 
end



