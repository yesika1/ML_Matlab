
% FEATURENORMALIZE Normalizes the features in X 
% FEATURENORMALIZE(X) returns a normalized version of X where
%  the mean value of each feature is 0 and the standard deviation is 1.

%  Performing feature scaling can make gradient descent converge much more quickly.

% ==============================================================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% ou can use the mean() and sigma() functions to get the mean and 
% std deviation for each column of X. These are returned as row vectors (1 x n)
% Now you want to apply those values to each element in every row of the X matrix. 
% One way to do this is to duplicate these vectors for each row in X, so they're the same size.
% One method to do this is to create a column vector of all-ones - size (m x 1) - and 
% multiply it by the mu or sigma row vector (1 x n). 
% Dimensionally, (m x 1) * (1 x n) gives you a (m x n) matrix, and every row of the resulting matrix will be identical.      
% Now that X, mu, and sigma are all the same size, you can use element-wise operators to compute X_normalized.
%
% ============================================================

function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X)              % returns a row vector
sigma = std(X)            % returns a row vector
m = size(X, 1)            % returns the number of rows in X
mu_matrix = ones(m, 1) * mu  
sigma_matrix = ones(m, 1) * sigma

X_norm = (X- mu_matrix)./sigma_matrix  % subtract the mu matrix from X, and divide element-wise by the sigma matrix


end
