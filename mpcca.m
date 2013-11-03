function [meanV, covarianceV, transformX, transformY] = mpcca(X, Y, dimZ, numComponents, maxIterations)
%MCCA apply mixture probabilistic CCA method on multi-view problems
% using EM algorithm
%INPUT:
% featureX (dimX x numSamples) : View 1 
% featureY (dimY x numSamples) : View 2
% dimZ : dimension of shared space for featureZ
% numComponents : number of Gaussian mixtures
% maxIterations : maximum number of iterations of EM (default 100)
% endFlag : termination tolerance (change in probability likelihood) (default 0.0001)
%INTERMEDIATE
% transformX (dimX x dimZ x numComponents) : transformation matrix from featureZ to featureX
% transformY (dimX x dimZ x numComponents) : transformation matrix from featureZ to featureY
% meanX (numComponents x dimX) :  mean for featureX
% meanY (numComponents x dimY) :  mean for featureY
% varNoiseX ( dimX x dimX x numComponents): covariance for noise in channel X
% varNoiseY ( dimY x dimY x numComponents): covariance for noise in channel Y
%OUTPUT
% meanV (dimV x numComponents) : mean for joint featureV
% covarianceV (dimV x dimV x numComponents) : covariance for joint featureV
% weight (numComponents x 1) :  prior for each component
%MODELS
% featureX = transformX * featureZ + meanX + noiseX
% featureY = transformY * featureZ + meanY + noiseY
%Progress Flag
% logLikelihood : log likelihood
%Termination Condition
% Iterates until a likelihood change < tolerance
% or iterations reach maxIterations

%% control randomness %%
rng('default') ;

%%%%%%%%%%%%%%%%%%%%
%% Initialization %%
%%%%%%%%%%%%%%%%%%%%
%% checking data validity
if size(X, 2) ~= size(Y, 2)
	error('multi-view features do not have same amount of data') ;
else
	% V = [X ; Y] ;
	numSamples = size([X ; Y], 2) ;
	dimV = size([X ; Y], 1) ;
end
dimX = size(X, 1) ;
dimY = size(Y, 1) ;

% [meanV, covarianceV, weight, logLikelihood] = vl_gmm([X ; Y], numComponents) ;
%% reserve non-diagonal element (currently)
% initCovarianceV = zeros(dimV, dimV, numComponents) ;
% for indexComponent = 1 : numComponents
	% initCovarianceV(:, :, indexComponent) = diag(covarianceV(:, indexComponent));
% end
% covarianceV = initCovarianceV ;
%% covariance of X and Y given Z
% covarianceX = covarianceV([1 : dimX], :) ;
% covarianceY = covarianceV([1 + dimX : end], :) ;

%% using Gaussian mixture model to initialize parameters %%
disp('***** applying Gaussian mixture modelling *****') ;
gmModel = gmdistribution.fit([X ; Y]', numComponents, 'Regularize', 0.0001) ;
meanV = gmModel.mu' ;
covarianceV = gmModel.Sigma ;
weight = gmModel.PComponents ;
%% set of parameters for E-step
meanX = meanV([1 : dimX], :) ;
meanY = meanV([1 + dimX : end], :) ;

%% using probabilistic cca model to initialize parameters %%
transformX = zeros(dimX, dimZ, numComponents) ;
transformY = zeros(dimY, dimZ, numComponents) ;
covarianceX = zeros(dimX, dimX, numComponents) ;
covarianceY = zeros(dimY, dimY, numComponents) ;
%% initial covariance of X and Y given Z and transform matrix
disp('***** applying canonical correlation modelling *****') ;
for indexComponent = 1 : numComponents
	[transformX(:, :, indexComponent), transformY(:, :, indexComponent), ...
	covarianceX(:, :, indexComponent), covarianceY(:, :, indexComponent), relation] = ...
	pcca(dimX, dimY, dimZ, covarianceV(:, :, indexComponent)) ;
	
	transformV_k = [transformX(:, :, indexComponent) ; transformY(:, :, indexComponent)] ;
	covarianceV(:, :, indexComponent) = transformV_k * transformV_k' ;
	covarianceV(1 : dimX, 1 : dimX, indexComponent) = ...
	covarianceV(1 : dimX, 1 : dimX, indexComponent) + covarianceX(:, :, indexComponent) ;
	covarianceV(1 + dimX : end, 1 + dimX : end, indexComponent) = ...
	covarianceV(1 + dimX : end, 1 + dimX : end, indexComponent) + covarianceY(:, :, indexComponent) ;
end

gmModel = gmdistribution(meanV', covarianceV, weight) ;
fprintf('Iteration#%03d: ', 0) ;
fprintf('logLikelihood = %d\n', sum(log(pdf(gmModel, [X ; Y]')))) ;
%%%%%%%%%%%%%%%%%%%%%%%
%% EM ALGORITHM LOOP %%
%%%%%%%%%%%%%%%%%%%%%%%
for indexIterate = 1 : maxIterations
	%% E-step %%
	% update Gamma_{i, k} : probability of sample x_i belonging to component k : Pr{Component J | point I}
	% Gamma : numSamples x numComponents 
	Gamma = posterior(gmModel, [X ; Y]') ; 			
	%% prior
	weight = sum(Gamma, 1) ;
	disp(weight / numSamples) ;
	
	for indexComponent = 1 : numComponents
		fprintf('\t EM iteration: %d\n', indexComponent) ;
		%% separately loop for each component %%
		
        %%%%%%%%%%%%
		%% E-step %%
		%%%%%%%%%%%%
		
		%% mean
		meanX(:, indexComponent) = X * Gamma(:, indexComponent) / weight(indexComponent);
		meanY(:, indexComponent) = Y * Gamma(:, indexComponent) / weight(indexComponent);
		
		%%%%%%%%%%%%
		%% M-step %%
		%%%%%%%%%%%%
		
		inverseRegul = 0 ;
% 		invCovarianceX_k = diag(1 ./ diag(covarianceX(:, :, indexComponent)) + ...
% 						inverseRegul * max(diag(covarianceX(:, :, indexComponent)))) ;
% 		invCovarianceY_k = diag(1 ./ diag(covarianceY(:, :, indexComponent)) + ...
% 						inverseRegul * max(diag(covarianceY(:, :, indexComponent)))) ;					
		%% transform matrix for each component
		transformX_k = transformX(:, :, indexComponent) ;
		transformY_k = transformY(:, :, indexComponent) ;

		%%%%%%%%%%%%%%%
		%% posterior %%
		%%%%%%%%%%%%%%%
		% Var(Z_k) : covariance of latent variable z for k-th component
		covarianceZ_k = eye(dimZ) - ...
			[transformX_k' transformY_k'] / ...
			(covarianceV(:, :, indexComponent) + inverseRegul * eye(dimV)) * ...
			[transformX_k ; transformY_k] ;			
		% E(Z_k) : latent Z samples, dimZ x numSamples
% 		EZ_k = covarianceZ_k * ...
% 			( transformX_k' / covarianceX(:, :, indexComponent) * bsxfun(@minus, X, meanX(:, indexComponent)) + ...
% 			  transformY_k' / covarianceY(:, :, indexComponent) * bsxfun(@minus, Y, meanY(:, indexComponent)) ) ;
        EZ_k = [transformX_k' transformY_k'] / covarianceV(:, : , indexComponent) * ...
            [bsxfun(@minus, X, meanX(:, indexComponent)) ; bsxfun(@minus, Y, meanY(:, indexComponent))] ;
        
		correlationZ_k = weight(indexComponent) * covarianceZ_k ;
		transformX_k = zeros(dimX, dimZ) ;
		transformY_k = zeros(dimY, dimZ) ;
		%%% parallel computing %%%
		for indexSample = 1 : numSamples
			% E(Z_kZ_k) : correlation of Z for k-th component
			correlationZ_k = correlationZ_k + ...
				Gamma(indexSample, indexComponent) * EZ_k(:, indexSample) * EZ_k(:, indexSample)' ;
			% W_x^k, W_y^k : transform matrix
			transformX_k = transformX_k + ...
				Gamma(indexSample, indexComponent) * (X(:, indexSample) - meanX(:, indexComponent)) * EZ_k(:, indexSample)' ;
			transformY_k = transformY_k + ...
				Gamma(indexSample, indexComponent) * (Y(:, indexSample) - meanY(:, indexComponent)) * EZ_k(:, indexSample)' ;
		end
		%% update transform models %%
		transformX_k = transformX_k / correlationZ_k ;
		transformY_k = transformY_k / correlationZ_k ;		
	    % transform matrix that map Z to X and Y
		transformX(:, :, indexComponent) = transformX_k ;
		transformY(:, :, indexComponent) = transformY_k ;

		centerX = bsxfun(@minus, X - transformX(:, :, indexComponent) * EZ_k, meanX(:, indexComponent)) ;
		centerY = bsxfun(@minus, Y - transformY(:, :, indexComponent) * EZ_k, meanY(:, indexComponent)) ;
		% covariance for X and Y : full
		covarianceX_k = transformX_k * covarianceZ_k * transformX_k' * weight(indexComponent) ;
		covarianceY_k = transformY_k * covarianceZ_k * transformY_k' * weight(indexComponent) ;		
		%%%% parallel computing %%%%
		for indexSample = 1 : numSamples
			covarianceX_k = covarianceX_k + ...
				Gamma(indexSample, indexComponent) * centerX(:, indexSample) * centerX(:, indexSample)' ;
			covarianceY_k = covarianceY_k + ...
				Gamma(indexSample, indexComponent) * centerY(:, indexSample) * centerY(:, indexSample)' ;
		end
		%% update noise models %%
		covarianceX_k = covarianceX_k / weight(indexComponent) ;
		covarianceY_k = covarianceY_k / weight(indexComponent) ;
		% covariance matrix of noise models
		covarianceX(:, :, indexComponent) = covarianceX_k ;
		covarianceY(:, :, indexComponent) = covarianceY_k ;
		
		% intermediate variable
		transformV_k = [transformX(:, :, indexComponent) ; transformY(:, :, indexComponent)] ;
		covarianceV(:, :, indexComponent) = transformV_k * transformV_k' ;
		covarianceV(1 : dimX, 1 : dimX, indexComponent) =  covarianceV(1 : dimX, 1 : dimX, indexComponent) + covarianceX_k ;
		covarianceV(1 + dimX : end, 1 + dimX : end, indexComponent) =  covarianceV(1 + dimX : end, 1 + dimX : end, indexComponent) + covarianceY_k ;
	end

	%% M-step %%
	weight = weight / numSamples ;
	meanV = [meanX ; meanY] ;
	gmModel = gmdistribution(meanV', covarianceV, weight) ;
	logLikelihood = sum(log(pdf(gmModel, [X ; Y]'))) ;
	fprintf('Iteration#%03d: logLikelihood = %d\n', indexIterate, logLikelihood) ;
end
end