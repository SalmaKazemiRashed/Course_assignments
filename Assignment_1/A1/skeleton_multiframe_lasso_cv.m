function [Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(T,X,lambdavec,K)
% [wopt,lambdaopt,VMSE,EMSE] = multiframe_lasso_cv(T,X,lambdavec,n)
% Calculates the LASSO solution for all frames and trains the
% hyperparameter using cross-validation.
%
%   Output:
%   Wopt        - mxnframes LASSO estimate for optimal lambda
%   lambdaopt   - optimal lambda value
%   VMSE        - vector of validation MSE values for lambdas in grid
%   EMSE        - vector of estimation MSE values for lambdas in grid
%
%   inputs:
%   T           - NNx1 data column vector
%   X           - NxM regression matrix
%   lambdavec   - vector grid of possible hyperparameters
%   K           - number of folds

% Define some sizes
NN = length(T);
[N,M] = size(X);
Nlam = length(lambdavec);
lambdaopt = [];


% Set indexing parameters for moving through the frames.
framehop = N;
idx = (1:N)';
framelocation = 0;
Nframes = 0;
tic
while framelocation + N <= NN
    Nframes = Nframes + 1; 
    framelocation = framelocation + framehop;
end % Calculate number of frames.
toc
% Preallocate
Wopt = zeros(M,Nframes);
SEval = zeros(K,Nlam);
SEest = zeros(K,Nlam);
MSEval = [];
MSEest = [];

% Set indexing parameter for the cross-validation indexing
Nval = floor(N/K);
cvhop = Nval;

%randomind = randperm(N);% Select random indices for picking out validation and estimation indices. 
framelocation = 0;
for kframe = 1:Nframes % First loop over frames
    
    cvlocation = 0;
    randomind = randperm(N);
    for kfold = 1:K % Then loop over the folds
    valind = randomind((kfold-1)*cvhop+1:(kfold)*cvhop) ;% Select validation indices
    allind = 1:N;
    allind(valind) = [];
    estind = allind; % Select estimation indices
    assert(isempty(intersect(valind,estind)), 'There are overlapping indices in valind and estind!'); % assert empty intersection between valind and estind
   
        
        
        t = T(framelocation + idx); % Set data in this frame
        wold = zeros(M,1);  % Initialize old weights for warm-starting.
        %tic
        
        test_w = [];
        for klam = 1:Nlam  % Finally loop over the lambda grid
            tic
           what = skeleton_lasso_ccd(t(estind),X(estind,:),lambdavec(klam),wold) ;
           toc
           % Calculate LASSO estimate on estimation indices for the current lambda-value.
           SEval(kfold,klam) = 1/Nval*(t(valind)-X(valind,:)*what)'* (t(valind)-X(valind,:)*what); % Calculate validation error for this estimate
           SEest(kfold,klam) = 1/(N-Nval)*(t(estind)-X(estind,:)*what)'* (t(estind)-X(estind,:)*what); % Calculate estimation error for this estimate
           wold = what; % Set current estimate as old estimate for next lambda-value.
            test_w  = [test_w , what];
            disp(['Frame: ' num2str(kframe) ', Fold: ' num2str(kfold) ', Hyperparam: ' num2str(klam)]) % Display progress through frames, folds and lambda-indices.
        end
        save('test_w')
       % toc
        cvlocation = cvlocation+cvhop; % Hop to location for next fold.
    end
    
    framelocation = framelocation + framehop;% Hop to location for next frame.
 
MSEval = [MSEval; mean(SEval,1)]; % Average validation error across folds
MSEest = [MSEest ; mean(SEest,1)]; % Average estimation error across folds
end
[val, ind] = min(mean(MSEval,1));
lambdaopt = lambdavec(ind);  % Select optimal lambda 

% Move through frames and calculate LASSO estimates using both estimation
% and validation data, store in Wopt.
framelocation = 0;

%for kframe = 1:Nframes
    %t = T(framelocation + idx);
   % Wopt(:,kframe) = skeleton_lasso_ccd(t,X,lambdaopt,wold);
   % framelocation = framelocation + framehop;
%end

RMSEval = sqrt(mean(MSEval,1));
RMSEest = sqrt(mean(MSEest,1));




end

